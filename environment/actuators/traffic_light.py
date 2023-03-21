from collections import defaultdict
from dataclasses import dataclass
from typing import cast, List, Optional, Union

import numpy as np
import traci
import gym

from .base_actuator import BaseActuator
from .traffic_light_util import determine_phases_set
from .definitions import METERS_PER_VEHICLE
from .lane_util import get_lane_leader, get_next_traffic_light_stats, get_short_lanes, get_inflowing_lanes


class TrafficLight(BaseActuator):
    @dataclass
    class Config(BaseActuator.Config):
        yellow_transition_duration: float = 2
        min_green_duration: float = 5
        cooldown_duration: float = 5  # Waiting time after an action until the traffic light accepts new actions
        reward_type: str = "time-spent-on-lanes"
        observations_add_more_statistics: bool = True  # Add more statistics-based obs, which SUMO-RL does not include
        observations_add_leading_vehicle: bool = False  # Add vehicle-specific obs (--> Vehicle-To-Infrastructure)
        enhance_short_lanes: bool = False  # Enhance obs & reward of *short lanes* w/ additional info on inflowing lanes

        def __post_init__(self):
            assert self.reward_type in ("sumo-rl-diff-waiting-time", "sumo-rl-average-speed", "sumo-rl-queue",
                                        "sumo-rl-pressure", "resco-wait-norm", "time-spent-on-lanes")
            assert self.yellow_transition_duration > 0
            assert self.min_green_duration > 0
            assert self.cooldown_duration > 0
            assert self.cooldown_duration > self.yellow_transition_duration

    def __init__(self, sumo_connection: traci.Connection, simulation_time: float, traffic_light_id: str, config: Config):
        super(TrafficLight, self).__init__(config=config)
        self.config = cast(TrafficLight.Config, self.config)
        self.traffic_light_id = traffic_light_id
        self._build_phases(sumo_connection=sumo_connection, config=config)
        self.reset(sumo_connection=sumo_connection, simulation_time=simulation_time)

        self._in_lanes = list(dict.fromkeys(sumo_connection.trafficlight.getControlledLanes(self.traffic_light_id)))
        self._in_lanes__lengths = {lane: sumo_connection.lane.getLength(lane) for lane in self._in_lanes}
        self._in_lanes__max_speeds = {lane: sumo_connection.lane.getMaxSpeed(lane) for lane in self._in_lanes}
        self._in_lanes__n_vehicles = \
            {lane: length/METERS_PER_VEHICLE for lane, length in self._in_lanes__lengths.items()}
        self._out_lanes = sorted(set(
            link[0][1] for link in sumo_connection.trafficlight.getControlledLinks(self.traffic_light_id) if link))

        if config.enhance_short_lanes is True:
            _short_lanes = [lane for lane in get_short_lanes(sumo_connection=sumo_connection) if lane in self._in_lanes]
            self._additional_lanes = []
            for _lane_id in _short_lanes:
                _inflowing = get_inflowing_lanes(sumo_connection, lane_id=_lane_id)
                self._additional_lanes.extend(_inflowing)
            self._additional_lanes = sorted(self._additional_lanes)

            self._additional_lanes__lengths = {l: sumo_connection.lane.getLength(l) for l in self._additional_lanes}
            self._additional_lanes__max_speeds = {l: sumo_connection.lane.getMaxSpeed(l) for l in self._additional_lanes}
            self._additional_lanes__n_vehicles = \
                {lane: length/METERS_PER_VEHICLE for lane, length in self._additional_lanes__lengths.items()}

        _observations = self.get_observations(sumo_connection=sumo_connection, simulation_time=simulation_time,
                                              vehicles__acc_lane_waiting_times=defaultdict(lambda: defaultdict(float)))
        self._observation_space = gym.spaces.Box(low=np.zeros_like(_observations), high=np.ones_like(_observations))
        self._action_space = gym.spaces.Discrete(len(self._green_phases))

    @staticmethod
    def construct_instances(sumo_connection: traci.Connection, simulation_time: float, actuator_config: Config,
                            **kwargs) -> List[BaseActuator]:
        assert isinstance(actuator_config, TrafficLight.Config)

        actuators = []
        for _id in sumo_connection.trafficlight.getIDList():
            _traffic_light = TrafficLight(sumo_connection=sumo_connection, simulation_time=simulation_time,
                                          traffic_light_id=_id, config=actuator_config)
            actuators += [_traffic_light]
        return actuators

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def id(self) -> str:
        return self.traffic_light_id

    @property
    def last_reward(self) -> Optional[float]:
        return self._last_reward

    def _build_phases(self, sumo_connection: traci.Connection, config: Config):
        _logic: traci.trafficlight.Logic = sumo_connection.trafficlight.getAllProgramLogics(self.traffic_light_id)[0]
        _phases: list[traci.trafficlight.Phase] = list(_logic.phases)
        assert all(len(p.state) == len(_phases[0].state) for p in _phases), \
            "Phase states are expected to be of equal length!"

        self._green_phases, self._yellow_transition_phases = determine_phases_set(
            sumo_phases=_phases, yellow_transition_duration=config.yellow_transition_duration)

        # Sort our freshly generated green-phases, so we don't end-up with a mess when reloading checkpointed models
        self._green_phases = sorted(self._green_phases, key=lambda p: p.state)

        _logic.type = 0
        _logic.phases = self._green_phases + [y for y in self._yellow_transition_phases.values()]
        sumo_connection.trafficlight.setProgramLogic(self.traffic_light_id, _logic)

    def _set_green_phase(self, sumo_connection: traci.Connection, simulation_time: float, green_phase_index: int):
        assert 0 <= green_phase_index < len(self._green_phases)

        _wait_time = self.config.yellow_transition_duration + self.config.min_green_duration
        if self._current_green_phase_index is None:  # This is only the case during reset()
            next_phase = self._green_phases[green_phase_index]
            self._current_green_phase_index = green_phase_index
        elif self._current_green_phase_index == green_phase_index or self._time_since_last_phase_change < _wait_time:
            next_phase = self._green_phases[self._current_green_phase_index]
            self._next_action_time = simulation_time + self.config.cooldown_duration
        else:
            _from_phase = self._green_phases[self._current_green_phase_index]
            _to_phase = self._green_phases[green_phase_index]
            next_phase = self._yellow_transition_phases[(_from_phase.state, _to_phase.state)]
            self._current_green_phase_index = green_phase_index
            self._is_yellow = True
            self._time_since_last_phase_change = 0
            self._next_action_time = simulation_time + self.config.cooldown_duration
        sumo_connection.trafficlight.setRedYellowGreenState(self.id, next_phase.state)

    def reset(self, sumo_connection: traci.Connection, simulation_time: float):
        self._is_yellow, self._current_green_phase_index = False, None
        self._set_green_phase(sumo_connection=sumo_connection, simulation_time=simulation_time, green_phase_index=0)

        self._time_since_last_phase_change = 0
        self._next_action_time = 0  # Ensures that this actuator accepts new actions right away
        assert self.accepts_new_actions(sumo_connection=sumo_connection, simulation_time=simulation_time) is True
        self._last_reward, self._last_lanes_waiting_time = None, 0

    def tick(self, sumo_connection: traci.Connection, simulation_time: float):
        self._time_since_last_phase_change += 1
        if self._is_yellow and self._time_since_last_phase_change >= self.config.yellow_transition_duration:
            sumo_connection.trafficlight.setRedYellowGreenState(self.id, self._green_phases[self._current_green_phase_index].state)
            self._is_yellow = False

    def accepts_new_actions(self, sumo_connection: traci.Connection, simulation_time: float) -> bool:
        return self._next_action_time <= simulation_time

    def act(self, sumo_connection: traci.Connection, simulation_time: float, action: Union[int, float, np.ndarray]):
        assert isinstance(action, int) or np.issubdtype(action, np.integer), f"Wrong action type: {type(action)}"
        self._set_green_phase(sumo_connection=sumo_connection, simulation_time=simulation_time, green_phase_index=action)

    def get_observations(self, sumo_connection: traci.Connection, simulation_time: float,
                         vehicles__acc_lane_waiting_times: dict[str, dict[str, float]]) -> np.ndarray:
        _l = sumo_connection.lane

        # The following default-observations are the same as in SUMO-RL
        phase_idx = [(1 if self._current_green_phase_index == i else 0) for i in range(len(self._green_phases))]
        min_green = [0 if self._time_since_last_phase_change <
                          self.config.min_green_duration + self.config.yellow_transition_duration else 1]
        lanes_density = [_l.getLastStepVehicleNumber(l) / self._in_lanes__n_vehicles[l] for l in self._in_lanes]
        lanes_queue = [_l.getLastStepHaltingNumber(l) / self._in_lanes__n_vehicles[l] for l in self._in_lanes]
        observations = phase_idx + min_green + lanes_density + lanes_queue

        # Optionally add more statistics-based obs
        if self.config.observations_add_more_statistics is True:
            # No need to include "n_approaching_vehicles" here, as it is a linear combination of "lanes_density"
            # and "lanes_queue".
            _waiting_time_per_lane = self._get_waiting_time_per_lane(
                sumo_connection=sumo_connection, vehicles__acc_lane_waiting_times=vehicles__acc_lane_waiting_times)
            state_duration = [self._time_since_last_phase_change / 60]
            lanes_waiting_time = [_waiting_time_per_lane[l] / (n * 10) for l, n in self._in_lanes__n_vehicles.items()]
            lanes_mean_speed = [0 if lanes_density[i] == 0 else _l.getLastStepMeanSpeed(lane_id) / max_speed
                                for i, (lane_id, max_speed) in enumerate(self._in_lanes__max_speeds.items())]
            observations += state_duration + lanes_waiting_time + lanes_mean_speed

        # Optionally add vehicle-based data (--> Vehicle-To-Infrastructure)
        if self.config.observations_add_leading_vehicle is True:
            distances_to_tl, distances_to_next_standing_vh, speeds = [], [], []
            for _lane_id in self._in_lanes:
                _lane_length = self._in_lanes__lengths[_lane_id]
                _leader_id, _leader_distance_to_vh = get_lane_leader(sumo_connection, lane_id=_lane_id)
                _leader_distance_to_tl, _leader_speed = 1, 0
                if _leader_id is not None:
                    _leader_speed = sumo_connection.vehicle.getSpeed(_leader_id) / self._in_lanes__max_speeds[_lane_id]
                    _stats = get_next_traffic_light_stats(sumo_connection, vehicle_id=_leader_id)
                    if _stats is not None:
                        _leader_distance_to_tl, _, _ = _stats
                        _leader_distance_to_tl /= _lane_length
                distances_to_tl += [_leader_distance_to_tl]
                distances_to_next_standing_vh += [_leader_distance_to_vh / _lane_length if _leader_distance_to_vh is not None else 1]
                speeds += [_leader_speed]
            observations += distances_to_tl + distances_to_next_standing_vh + speeds

        # Optionally enhance short lanes, by incorporating lanes that lead to short lanes ("additional lanes")
        if self.config.enhance_short_lanes is True:
            _n_vehicles = {lane_id: _l.getLastStepVehicleNumber(lane_id) for lane_id in self._additional_lanes}
            additional_density = [_n_vehicles[l] / self._additional_lanes__n_vehicles[l] for l in self._additional_lanes]
            additional_mean_speed = \
                [(0 if _n_vehicles[l] == 0 else sumo_connection.lane.getLastStepMeanSpeed(l)/self._additional_lanes__max_speeds[l])
                 for l in self._additional_lanes]
            observations += additional_density + additional_mean_speed

        observations = np.array(observations, dtype=np.float32).clip(min=0, max=1)
        return observations

    def get_reward(self, sumo_connection: traci.Connection, simulation_time: float,
                   vehicles__acc_lane_waiting_times: dict[str, dict[str, float]],
                   lanes__vehicle_usage_times: dict[str, dict[str, float]]) -> float:
        _reward_type = self.config.reward_type
        if _reward_type == "sumo-rl-diff-waiting-time":
            _waiting_time_per_lane = self._get_waiting_time_per_lane(
                sumo_connection=sumo_connection, vehicles__acc_lane_waiting_times=vehicles__acc_lane_waiting_times)
            _n_lanes = len(_waiting_time_per_lane)
            tl_wait = sum(_waiting_time_per_lane.values()) / (_n_lanes * 25)
            self._last_reward = self._last_lanes_waiting_time - tl_wait
            self._last_lanes_waiting_time = tl_wait
        elif _reward_type == "sumo-rl-average-speed":
            self._last_reward = self._get_average_speed(sumo_connection=sumo_connection)
        elif _reward_type == "sumo-rl-queue":
            self._last_reward = -self.get_total_queued(sumo_connection=sumo_connection)
        elif _reward_type == "sumo-rl-pressure":
            self._last_reward = -self._get_pressure(sumo_connection=sumo_connection)
        elif _reward_type == "resco-wait-norm":  # This reward fn equals "clipped-waiting-time" from Lane/LaneCompound
            _waiting_time_per_lane = self._get_waiting_time_per_lane(
                sumo_connection=sumo_connection, vehicles__acc_lane_waiting_times=vehicles__acc_lane_waiting_times)
            _n_lanes = len(_waiting_time_per_lane)
            tl_wait = sum(_waiting_time_per_lane.values()) / (_n_lanes * 56)
            self._last_reward = -float(np.clip(tl_wait, -4, 4).astype(np.float32))
        elif _reward_type == "time-spent-on-lanes":
            _time_spent_per_lane: dict[str, float] = self._get_time_spent_per_lane_norm(lanes__vehicle_usage_times)
            _n_lanes = len(_time_spent_per_lane)
            time_spent = sum(_time_spent_per_lane.values()) / _n_lanes
            self._last_reward = -time_spent / 50
        else:
            raise RuntimeError("We should not have ended up here!")
        return self._last_reward

    def _get_waiting_time_per_lane(
            self, sumo_connection: traci.Connection,
            vehicles__acc_lane_waiting_times: dict[str, dict[str, float]]) -> dict[str, float]:
        wait_time_per_lane = {}
        _lanes = self._in_lanes if self.config.enhance_short_lanes is False else (self._in_lanes + self._additional_lanes)
        for lane_id in _lanes:
            vehicles = sumo_connection.lane.getLastStepVehicleIDs(lane_id)
            wait_time = sum([vehicles__acc_lane_waiting_times[v][lane_id] for v in vehicles])
            wait_time_per_lane[lane_id] = wait_time
        return wait_time_per_lane

    def _get_time_spent_per_lane_norm(self, lanes__vehicle_usage_times: dict[str, dict[str, float]]) -> dict[str, float]:
        _lanes__n_vehicles = self._in_lanes__n_vehicles if self.config.enhance_short_lanes is False else \
            dict(self._in_lanes__n_vehicles, **self._additional_lanes__n_vehicles)
        time_spent = {l: sum(lanes__vehicle_usage_times[l].values())/n_veh for l, n_veh in _lanes__n_vehicles.items()}
        return time_spent

    def _get_average_speed(self, sumo_connection: traci.Connection) -> float:
        _vehicles = []
        _lanes = self._in_lanes if self.config.enhance_short_lanes is False else (self._in_lanes+self._additional_lanes)
        for lane in _lanes:
            _vehicles += sumo_connection.lane.getLastStepVehicleIDs(lane)
        if len(_vehicles) == 0:
            return 1.0
        avg_speed = 0.0
        for v in _vehicles:
            avg_speed += sumo_connection.vehicle.getSpeed(v) / sumo_connection.vehicle.getAllowedSpeed(v)
        return avg_speed / len(_vehicles)

    def get_total_queued(self, sumo_connection: traci.Connection) -> int:
        _lanes = self._in_lanes if self.config.enhance_short_lanes is False else (self._in_lanes+self._additional_lanes)
        return sum(sumo_connection.lane.getLastStepHaltingNumber(lane) for lane in _lanes)

    def _get_pressure(self, sumo_connection: traci.Connection) -> float:
        return sum(sumo_connection.lane.getLastStepVehicleNumber(lane) for lane in self._out_lanes) - \
               sum(sumo_connection.lane.getLastStepVehicleNumber(lane) for lane in self._in_lanes)

    def get_infos(self, sumo_connection: traci.Connection, simulation_time: float) -> dict[str, Union[float, int]]:
        infos = {
            "reward": self._last_reward,
            "average-speed": self._get_average_speed(sumo_connection=sumo_connection),
            "queue": self.get_total_queued(sumo_connection=sumo_connection),
            "pressure": self._get_pressure(sumo_connection=sumo_connection)
        }
        return infos

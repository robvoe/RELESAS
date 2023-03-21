from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional, cast, Tuple, Union, Dict, Literal

import gym
import numpy as np
import traci

from .base_actuator import BaseActuator
from .definitions import METERS_PER_VEHICLE, DEFAULT_MAX_VEHICLE_SPEED
from .lane_util import get_lane_leader, get_next_traffic_light_stats, get_speed_controllable_lanes, \
    cluster_lanes_by_traffic_light, cluster_lanes_by_edge, get_lane_specific_traffic_light
from .lane import Lane, MIN_SPEED_FACTOR
from util.embedding import OneHotEmbedding


class LaneCompound(BaseActuator):
    """Very similar to Lane actuator, only that it controls a bunch of lanes, in order to lower train cost."""
    @dataclass
    class Config(BaseActuator.Config):
        cluster_lanes_by: Literal["traffic-light", "edge"] = "traffic-light"
        reward_type: str = "time-spent-on-lanes"
        cooldown_duration: float = 5  # Waiting time after an action until street accepts new actions. Speeds up train
        subject_to_speed_limit: Literal["leading-vehicle", "whole-lane"] = "leading-vehicle"
        leading_vehicle_consider_individual_speed_factor: bool = False
        observations_consider_leading_vehicle: bool = True
        physical_prediction_measure: str = "speed-change"

        def __post_init__(self):
            # Transform legacy values. Necessary to run old experiments
            if self.physical_prediction_measure == "acceleration":
                self.physical_prediction_measure = "speed-change"
            elif self.physical_prediction_measure == "velocity":
                self.physical_prediction_measure = "speed"

            # Value checks
            assert self.cluster_lanes_by in ("traffic-light", "edge")
            assert self.reward_type in \
                   ("average-speed", "diff-waiting-time", "clipped-waiting-time",
                    "diff-total-co2-emission", "total-co2-emission", "smoothed-total-co2-emission",
                    "time-spent-on-lanes")
            assert self.subject_to_speed_limit in ("whole-lane", "leading-vehicle")
            assert self.physical_prediction_measure in ("speed", "speed-change")
            assert self.cooldown_duration >= 1

    def __init__(self, sumo_connection: traci.Connection, simulation_time: float, lane_ids: list[str], config: Config):
        assert len(lane_ids) > 0
        assert len(lane_ids) == len(set(lane_ids)), "No duplicate lane ids allowed!"

        super(LaneCompound, self).__init__(config=config)
        self.config = cast(Lane.Config, self.config)

        # Make sure all lanes lead to the same TL
        _lane_ids__tl_ids = get_lane_specific_traffic_light(sumo_connection, lane_ids=lane_ids)
        _tl_ids = set(_lane_ids__tl_ids.values())
        assert len(_tl_ids) == 1, "All given lanes must lead to a single mutual traffic light!"
        self._mutual_traffic_light_id: str = next(iter(_tl_ids))
        assert self._mutual_traffic_light_id is not None

        self._lane_ids = lane_ids = sorted(lane_ids)
        self._lane_ids__lengths = {lane: sumo_connection.lane.getLength(lane) for lane in lane_ids}
        self._lane_ids__max_speed = {lane: sumo_connection.lane.getMaxSpeed(lane) for lane in lane_ids}
        self._lane_ids__n_vehicles = {lane: length/METERS_PER_VEHICLE for lane, length in self._lane_ids__lengths.items()}

        self.reset(sumo_connection=sumo_connection, simulation_time=simulation_time)
        self._handle_speed_limit(sumo_connection=sumo_connection)

        # Build the embedding to represent the TL states
        _logic: traci.trafficlight.Logic = \
            sumo_connection.trafficlight.getAllProgramLogics(self._mutual_traffic_light_id)[0]
        _tl_states: list[str] = sorted(set(phase.state for phase in _logic.phases))
        self._tl_state_embedding = OneHotEmbedding(_tl_states)

        # Determine action & observation spaces
        _observations = self.get_observations(sumo_connection=sumo_connection, simulation_time=simulation_time,
                                              vehicles__acc_lane_waiting_times=defaultdict(lambda: defaultdict(float)))
        self._observation_space = gym.spaces.Box(low=np.zeros_like(_observations), high=np.ones_like(_observations)*2)

        self._action_shape = _shape = (len(lane_ids), )
        if config.physical_prediction_measure == "speed":
            self._action_space = gym.spaces.Box(low=np.float(MIN_SPEED_FACTOR), high=np.float(1),
                                                shape=_shape, dtype=np.float32)
        elif config.physical_prediction_measure == "speed-change":
            self._action_space = gym.spaces.Box(low=np.float(-0.1), high=np.float(0.1), shape=_shape, dtype=np.float32)
        else:
            raise RuntimeError("We should not have ended-up here!")

    @staticmethod
    def construct_instances(sumo_connection: traci.Connection, simulation_time: float, actuator_config: Config,
                            **kwargs) -> List[BaseActuator]:
        assert isinstance(actuator_config, LaneCompound.Config)

        _lane_ids = get_speed_controllable_lanes(sumo_connection=sumo_connection, filter_edge_id=None)
        assert len(_lane_ids) > 0, \
            "No edges left after filtering. Perhaps sth wrong w/ filter rules in function above?"

        if actuator_config.cluster_lanes_by == "traffic-light":
            _tl_ids__lane_ids = cluster_lanes_by_traffic_light(sumo_connection, lane_ids=_lane_ids)
            _clustered_lane_lists = list(_tl_ids__lane_ids.values())
        elif actuator_config.cluster_lanes_by == "edge":
            _edge_ids__lane_ids = cluster_lanes_by_edge(sumo_connection, lane_ids=_lane_ids)
            _clustered_lane_lists = list(_edge_ids__lane_ids.values())
        else:
            raise RuntimeError("We should not have ended-up here!")

        actuators = []
        for _lane_ids in _clustered_lane_lists:
            _lane_compound = LaneCompound(sumo_connection=sumo_connection, simulation_time=simulation_time,
                                          lane_ids=_lane_ids, config=actuator_config)
            actuators += [_lane_compound]
        return actuators

    @property
    def id(self) -> str:
        return self._mutual_traffic_light_id

    @property
    def controlled_lane_ids(self) -> list[str]:
        return self._lane_ids

    @property
    def last_reward(self) -> Optional[float]:
        return self._last_reward

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def current_speed_limits_norm(self) -> dict[str, float]:
        """
        Returns the current speed limit of all speed-controlled lanes, as values in the range [0; 1].

        @return: A dictionary that maps SUMO-lane-IDs to their respective normalized current speed limit.
        """
        return {lane_id: speed_limit / self._lane_ids__max_speed[lane_id]
                for lane_id, speed_limit in self._lane_ids__current_speed_limit.items()}

    def reset(self, sumo_connection: traci.Connection, simulation_time: float) -> None:
        self._lane_ids__last_leading_vehicle_id: dict[str, Optional[str]] = {l: None for l in self._lane_ids}
        self._lane_ids__current_leading_vehicle_id: dict[str, Optional[str]] = {l: None for l in self._lane_ids}
        self._lane_ids__current_leading_vehicle_id_distance: dict[str, Optional[float]] = {l: None for l in self._lane_ids}
        self._lane_ids__current_leading_vehicle_speed_factor: dict[str, Optional[float]] = {l: None for l in self._lane_ids}

        self._lane_ids__current_speed_limit = {l: self._lane_ids__max_speed[l] for l in self._lane_ids}
        self._last_reward, self._last_waiting_time, self._last_co2_emission = None, 0, 0
        self._co2_emission_memory: deque[float] = deque([0.0] * 5, maxlen=5)
        self._last_action_time = simulation_time - self.config.cooldown_duration

    def _handle_speed_limit(self, sumo_connection: traci.Connection) -> None:
        last_leader_ids = self._lane_ids__last_leading_vehicle_id
        current_leader_ids = self._lane_ids__current_leading_vehicle_id
        current_speed_limits = self._lane_ids__current_speed_limit
        current_leading_vehicle_speed_factors = self._lane_ids__current_leading_vehicle_speed_factor

        for lane_id in self._lane_ids:
            if self.config.subject_to_speed_limit == "leading-vehicle":
                _last_leader_id, _current_leader_id = last_leader_ids[lane_id], current_leader_ids[lane_id]
                _current_speed_limit = current_speed_limits[lane_id]
                if _last_leader_id is not None and \
                        (_current_leader_id != _last_leader_id or _current_leader_id is None):
                    # The previous leader does not lead (or is no part of the lane) anymore. Let's reset its max speed
                    if _last_leader_id not in sumo_connection.simulation.getArrivedIDList():
                        try:
                            sumo_connection.vehicle.setSpeed(_last_leader_id, -1)
                        except traci.TraCIException:
                            pass
                if _current_leader_id is not None:
                    _speed = current_speed_limits[lane_id] if not self.config.leading_vehicle_consider_individual_speed_factor \
                        else current_speed_limits[lane_id] * current_leading_vehicle_speed_factors[lane_id]
                    try:
                        sumo_connection.vehicle.setSpeed(_current_leader_id, _speed)
                    except traci.TraCIException:
                        pass
            elif self.config.subject_to_speed_limit == "whole-lane":
                sumo_connection.lane.setMaxSpeed(lane_id, current_speed_limits[lane_id])
            else:
                raise RuntimeError("We should not have ended-up here!")

    def tick(self, sumo_connection: traci.Connection, simulation_time: float) -> None:
        # Update leading vehicle infos
        for _l in self._lane_ids:
            self._lane_ids__last_leading_vehicle_id[_l] = self._lane_ids__current_leading_vehicle_id[_l]
            self._lane_ids__current_leading_vehicle_id[_l], self._lane_ids__current_leading_vehicle_id_distance[_l] = \
                get_lane_leader(sumo_connection, lane_id=_l)
            _current_leading_vehicle_id = self._lane_ids__current_leading_vehicle_id[_l]
            if _current_leading_vehicle_id != self._lane_ids__last_leading_vehicle_id[_l] and _current_leading_vehicle_id is not None:
                self._lane_ids__current_leading_vehicle_speed_factor[_l] = sumo_connection.vehicle.getSpeedFactor(_current_leading_vehicle_id)
            
        # Take care of the speed limit, if necessary
        if self.config.subject_to_speed_limit == "leading-vehicle":
            self._handle_speed_limit(sumo_connection=sumo_connection)

    def accepts_new_actions(self, sumo_connection: traci.Connection, simulation_time: float) -> bool:
        if simulation_time < self._last_action_time + self.config.cooldown_duration:
            return False
        return True

    def act(self, sumo_connection: traci.Connection, simulation_time: float, action: Union[int, float, np.ndarray]):
        assert isinstance(action, np.ndarray), f"Wrong action type: {type(action)}"
        assert action.shape == self._action_shape and action.dtype in (np.float, np.float32)

        for i, (_lane_id, _action) in enumerate(zip(self._lane_ids, action)):
            _max_speed = self._lane_ids__max_speed[_lane_id]
            _current_speed_limit = self._lane_ids__current_speed_limit[_lane_id]
            if self.config.physical_prediction_measure == "speed":
                assert 0.3 <= _action <= 1, f"Action value '{_action}' for lane #{i} ('{_lane_id}') out of bounds!"
                self._lane_ids__current_speed_limit[_lane_id] = _max_speed * _action
            elif self.config.physical_prediction_measure == "speed-change":
                assert -0.11 <= _action <= 0.11, f"Action value '{_action}' for lane #{i} ('{_lane_id}') out of bounds!"
                _current_speed_limit += _current_speed_limit * _action
                self._lane_ids__current_speed_limit[_lane_id] = np.clip(_current_speed_limit, 0.3*_max_speed, _max_speed)
            else:
                raise RuntimeError("We should not have ended-up here!")
        self._handle_speed_limit(sumo_connection=sumo_connection)
        self._last_action_time = simulation_time

    def get_observations(self, sumo_connection: traci.Connection, simulation_time: float,
                         vehicles__acc_lane_waiting_times: dict[str, dict[str, float]]) -> np.ndarray:
        # Lane statistics
        _tl_state_str = sumo_connection.trafficlight.getRedYellowGreenState(self._mutual_traffic_light_id)
        tl_state = self._tl_state_embedding(_tl_state_str)
        tl_state_duration = [sumo_connection.trafficlight.getPhaseDuration(self._mutual_traffic_light_id)/60]

        mean_speed, traffic_density, queue, current_speed_limit = [], [], [], []
        for _lane_id in self._lane_ids:
            _current_n_vehicles = sumo_connection.lane.getLastStepVehicleNumber(_lane_id)
            _n_vehicles, _max_speed = self._lane_ids__n_vehicles[_lane_id], self._lane_ids__max_speed[_lane_id]
            mean_speed += [0] if _current_n_vehicles == 0 else \
                [sumo_connection.lane.getLastStepMeanSpeed(_lane_id) / _max_speed]
            traffic_density += [_current_n_vehicles / _n_vehicles]
            queue += [sumo_connection.lane.getLastStepHaltingNumber(_lane_id) / _n_vehicles]
            current_speed_limit += [self._lane_ids__current_speed_limit[_lane_id] / _max_speed]

        observations = tl_state + mean_speed + traffic_density + queue + tl_state_duration + current_speed_limit

        # Vehicle-based data
        if self.config.observations_consider_leading_vehicle is True:
            for _lane_id in self._lane_ids:
                _length, _max_speed = self._lane_ids__lengths[_lane_id], self._lane_ids__max_speed[_lane_id]
                _current_leading_vehicle_id = self._lane_ids__current_leading_vehicle_id[_lane_id]
                _distance_to_vh = self._lane_ids__current_leading_vehicle_id_distance[_lane_id]

                _distance_to_tl, leading_vehicle_speed = _length, [0]
                if _current_leading_vehicle_id is not None:
                    _stats = get_next_traffic_light_stats(sumo_connection, vehicle_id=_current_leading_vehicle_id)
                    if _stats is not None:
                        _distance_to_tl, _, _traffic_light_id = _stats
                        assert _traffic_light_id == self._mutual_traffic_light_id
                    leading_vehicle_speed = [sumo_connection.vehicle.getSpeed(_current_leading_vehicle_id) / _max_speed]

                distance_to_traffic_light = [_distance_to_tl / _length]
                distance_to_next_standing_vehicle = [_distance_to_vh / _length if _distance_to_vh is not None else 1]
                observations += leading_vehicle_speed + distance_to_traffic_light + distance_to_next_standing_vehicle

        observations = np.array(observations, dtype=np.float32)
        return observations

    def get_reward(self, sumo_connection: traci.Connection, simulation_time: float,
                   vehicles__acc_lane_waiting_times: dict[str, dict[str, float]],
                   lanes__vehicle_usage_times: dict[str, dict[str, float]]) -> float:
        _lane_rewards = [self._get_reward_per_lane(sumo_connection, lane_id=l,
                                                   vehicles__acc_lane_waiting_times=vehicles__acc_lane_waiting_times,
                                                   lanes__vehicle_usage_times=lanes__vehicle_usage_times)
                         for l in self._lane_ids]
        _lane_rewards_mean = sum(_lane_rewards) / len(_lane_rewards)

        if self.config.reward_type in ("average-speed", "clipped-waiting-time", "total-co2-emission",
                                       "time-spent-on-lanes"):
            self._last_reward = _lane_rewards_mean
        elif self.config.reward_type in ("diff-waiting-time", "diff-total-co2-emission"):
            self._last_reward = _lane_rewards_mean - self._last_waiting_time
            self._last_waiting_time = _lane_rewards_mean
        elif self.config.reward_type in ("smoothed-total-co2-emission", ):
            self._co2_emission_memory.append(_lane_rewards_mean)
            smoothed = sum(v for v in self._co2_emission_memory) / len(self._co2_emission_memory)
            self._last_reward = smoothed
        else:
            raise RuntimeError("We should not have ended-up here!")
        return self._last_reward

    def _get_reward_per_lane(self, sumo_connection: traci.Connection, lane_id: str,
                             vehicles__acc_lane_waiting_times: dict[str, dict[str, float]],
                             lanes__vehicle_usage_times: dict[str, dict[str, float]]) -> float:
        if self.config.reward_type == "average-speed":
            _speed = sumo_connection.lane.getLastStepMeanSpeed(lane_id)
            return _speed / self._lane_ids__max_speed[lane_id]
        elif self.config.reward_type == "diff-waiting-time":
            waiting_time = Lane._get_waiting_time(sumo_connection=sumo_connection, lane_id=lane_id,
                                                  vehicles__lane_waiting_times=vehicles__acc_lane_waiting_times) / 100.0
            return -waiting_time
        elif self.config.reward_type == "clipped-waiting-time":  # This reward equals "resco-wait-norm" in TrafficLight
            waiting_time = Lane._get_waiting_time(sumo_connection=sumo_connection, lane_id=lane_id,
                                                  vehicles__lane_waiting_times=vehicles__acc_lane_waiting_times) / 100.0
            return -float(np.clip(waiting_time, -1, 1))
        elif self.config.reward_type == "diff-total-co2-emission":
            _n_vehicles = self._lane_ids__n_vehicles[lane_id]
            total_co2_emission = sumo_connection.lane.getCO2Emission(lane_id) / _n_vehicles
            return -total_co2_emission / 1000
        elif self.config.reward_type in ("total-co2-emission", "smoothed-total-co2-emission"):
            _n_vehicles = self._lane_ids__n_vehicles[lane_id]
            total_co2_emission = sumo_connection.lane.getCO2Emission(lane_id) / _n_vehicles
            return -total_co2_emission / 3000
        elif self.config.reward_type == "time-spent-on-lanes":
            _n_vehicles = self._lane_ids__n_vehicles[lane_id]
            time_spent_on_lane = Lane._get_time_spent_on_lane(lane_id, lanes__vehicle_usage_times)
            return -time_spent_on_lane / 50 / _n_vehicles
        else:
            raise RuntimeError("We should not have ended-up here!")

    def get_infos(self, sumo_connection: traci.Connection, simulation_time: float) -> dict[str, float]:
        return {}

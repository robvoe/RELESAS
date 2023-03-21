from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional, cast, Tuple, Union, Dict, Literal

import gym
import numpy as np
import traci

from .base_actuator import BaseActuator
from .definitions import METERS_PER_VEHICLE, DEFAULT_MAX_VEHICLE_SPEED
from .lane_util import get_lane_leader, get_next_traffic_light_stats, get_speed_controllable_lanes
from .traffic_light_util import traffic_light_state_to_scalars


MIN_SPEED_FACTOR = 0.3


class Lane(BaseActuator):
    @dataclass
    class Config(BaseActuator.Config):
        reward_type: str = "time-spent-on-lanes"
        cooldown_duration: float = 5  # Waiting time after an action until the lane accepts new actions. Speeds up train
        subject_to_speed_limit: Literal["leading-vehicle", "whole-lane"] = "leading-vehicle"
        leading_vehicle_consider_individual_speed_factor: bool = False
        observations_add_neighboring_lanes: bool = True
        observations_consider_leading_vehicle: bool = True
        physical_prediction_measure: str = "speed-change"

        def __post_init__(self):
            # Transform legacy values. Necessary to run old experiments
            if self.physical_prediction_measure == "acceleration":
                self.physical_prediction_measure = "speed-change"
            elif self.physical_prediction_measure == "velocity":
                self.physical_prediction_measure = "speed"

            # Value checks
            assert self.reward_type in \
                   ("average-speed", "diff-waiting-time", "clipped-waiting-time",
                    "diff-total-co2-emission", "total-co2-emission", "smoothed-total-co2-emission",
                    "time-spent-on-lanes")
            assert self.subject_to_speed_limit in ("whole-lane", "leading-vehicle")
            assert self.physical_prediction_measure in ("speed", "speed-change")
            assert self.cooldown_duration >= 1

    def __init__(self, sumo_connection: traci.Connection, simulation_time: float, lane_id: str, config: Config):
        super(Lane, self).__init__(config=config)
        self.config = cast(Lane.Config, self.config)

        self._lane_id = lane_id
        self._length = sumo_connection.lane.getLength(lane_id)
        self._max_speed = sumo_connection.lane.getMaxSpeed(lane_id)
        self._n_vehicles = self._length / METERS_PER_VEHICLE

        self.reset(sumo_connection=sumo_connection, simulation_time=simulation_time)
        self._handle_speed_limit(sumo_connection=sumo_connection)

        # Get infos about the associated (i.e. following) traffic light
        self._associated_traffic_light_id: str = self._get_associated_traffic_light(sumo_connection, lane_id=lane_id)
        assert self._associated_traffic_light_id is not None

        # Get neighboring lane ids
        _edge_id = sumo_connection.lane.getEdgeID(lane_id)
        _relevant_edge_lanes = sorted(get_speed_controllable_lanes(sumo_connection=sumo_connection, filter_edge_id=_edge_id))
        assert len(_relevant_edge_lanes) > 0, "Above function must return >= 1 lane as long it's called w/ valid edges!"
        assert lane_id in _relevant_edge_lanes
        self._neighboring_lane_ids = sorted(_relevant_edge_lanes)
        self._neighboring_lane_ids.remove(lane_id)

        # Below there is an experiment to apply some sorting to neighboring lanes, as an attempt to improve learning
        # self._neighboring_lane_ids = []
        # _my_lane_index = _relevant_edge_lanes.index(lane_id)
        # for i in range(0, max(_my_lane_index, len(_relevant_edge_lanes)-_my_lane_index-1)):
        #     _index_offset = i + 1
        #     if _my_lane_index - _index_offset >= 0:
        #         self._neighboring_lane_ids += [_relevant_edge_lanes[_my_lane_index - _index_offset]]
        #     if _my_lane_index + _index_offset <= len(_relevant_edge_lanes)-1:
        #         self._neighboring_lane_ids += [_relevant_edge_lanes[_my_lane_index + _index_offset]]

        # Determine action & observation spaces
        _observations = self.get_observations(sumo_connection=sumo_connection, simulation_time=simulation_time,
                                              vehicles__acc_lane_waiting_times=defaultdict(lambda: defaultdict(float)))
        self._observation_space = gym.spaces.Box(low=np.zeros_like(_observations), high=np.ones_like(_observations)*2)

        if config.physical_prediction_measure == "speed":
            self._action_space = gym.spaces.Box(low=np.float(MIN_SPEED_FACTOR), high=np.float(1),
                                                shape=(1, ), dtype=np.float32)
        elif config.physical_prediction_measure == "speed-change":
            self._action_space = gym.spaces.Box(low=np.float(-0.1), high=np.float(0.1), shape=(1,), dtype=np.float32)
        else:
            raise RuntimeError("We should not have ended-up here!")

    @staticmethod
    def construct_instances(sumo_connection: traci.Connection, simulation_time: float, actuator_config: Config,
                            **kwargs) -> List[BaseActuator]:
        assert isinstance(actuator_config, Lane.Config)

        actuators = []
        _lane_ids = get_speed_controllable_lanes(sumo_connection=sumo_connection, filter_edge_id=None)
        assert len(_lane_ids) > 0, "No lanes left after filtering. Perhaps sth wrong w/ filter rules in function above?"
        for _id in _lane_ids:
            _lane = Lane(sumo_connection=sumo_connection, simulation_time=simulation_time,
                         lane_id=_id, config=actuator_config)
            actuators += [_lane]
        return actuators

    @staticmethod
    def _get_associated_traffic_light(sumo_connection: traci.Connection, lane_id: str) -> Optional[str]:
        """Returns the id of the associated (i.e. following) traffic light, if available."""
        for _tl_id in sumo_connection.trafficlight.getIDList():
            if any(_l == lane_id for _l in sumo_connection.trafficlight.getControlledLanes(_tl_id)):
                return _tl_id
        return None

    @property
    def id(self) -> str:
        return self._lane_id

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
    def current_speed_limit_norm(self) -> float:
        """Returns the current speed limit as a value in the range [0; 1]."""
        return self._current_speed_limit / self._max_speed

    def reset(self, sumo_connection: traci.Connection, simulation_time: float) -> None:
        self._last_leading_vehicle_id: Optional[str] = None
        self._current_leading_vehicle_id: Optional[str] = None
        self._current_leading_vehicle_id_distance: Optional[float] = None
        self._current_leading_vehicle_speed_factor: Optional[float] = None

        self._current_speed_limit = self._max_speed
        self._last_reward, self._last_waiting_time, self._last_co2_emission = None, 0, 0
        self._co2_emission_memory: deque[float] = deque([0.0] * 5, maxlen=5)
        self._last_action_time = simulation_time - self.config.cooldown_duration

    def _handle_speed_limit(self, sumo_connection: traci.Connection) -> None:
        _last_leader_id, _current_leader_id = self._last_leading_vehicle_id, self._current_leading_vehicle_id

        if self.config.subject_to_speed_limit == "leading-vehicle":
            if _last_leader_id is not None and \
                    (_current_leader_id != _last_leader_id or _current_leader_id is None):
                # The previous leader does not lead (or is no part of the lane) anymore. Let's reset its max speed
                if _last_leader_id not in sumo_connection.simulation.getArrivedIDList():
                    sumo_connection.vehicle.setSpeed(_last_leader_id, -1)
            if _current_leader_id is not None:
                _speed = self._current_speed_limit if not self.config.leading_vehicle_consider_individual_speed_factor \
                    else self._current_speed_limit * self._current_leading_vehicle_speed_factor
                sumo_connection.vehicle.setSpeed(_current_leader_id, _speed)
        elif self.config.subject_to_speed_limit == "whole-lane":
            sumo_connection.lane.setMaxSpeed(self._lane_id, self._current_speed_limit)
        else:
            raise RuntimeError("We should not have ended-up here!")

    def tick(self, sumo_connection: traci.Connection, simulation_time: float) -> None:
        # Update leading vehicle infos
        self._last_leading_vehicle_id = self._current_leading_vehicle_id
        self._current_leading_vehicle_id, self._current_leading_vehicle_id_distance = \
            get_lane_leader(sumo_connection, lane_id=self._lane_id)
        if self._current_leading_vehicle_id != self._last_leading_vehicle_id and self._current_leading_vehicle_id is not None:
            self._current_leading_vehicle_speed_factor = sumo_connection.vehicle.getSpeedFactor(self._current_leading_vehicle_id)

        # Take care of the speed limit, if necessary
        if self.config.subject_to_speed_limit == "leading-vehicle":
            self._handle_speed_limit(sumo_connection=sumo_connection)

    def accepts_new_actions(self, sumo_connection: traci.Connection, simulation_time: float) -> bool:
        if simulation_time < self._last_action_time + self.config.cooldown_duration:
            return False
        return True

    def act(self, sumo_connection: traci.Connection, simulation_time: float, action: Union[int, float, np.ndarray]):
        assert isinstance(action, float) or isinstance(action, np.ndarray), f"Wrong action type: {type(action)}"
        if isinstance(action, np.ndarray):
            assert action.shape == (1, ) and action.dtype in (np.float, np.float32)
            action = action[0]
        if self.config.physical_prediction_measure == "speed":
            assert 0.3 <= action <= 1, f"Action value '{action}' out of bounds!"
            self._current_speed_limit = self._max_speed * action
        elif self.config.physical_prediction_measure == "speed-change":
            assert -0.11 <= action <= 0.11, f"Action value '{action}' out of bounds!"
            self._current_speed_limit += self._current_speed_limit * action
            self._current_speed_limit = np.clip(self._current_speed_limit, 0.3*self._max_speed, self._max_speed)
        else:
            raise RuntimeError("We should not have ended-up here!")
        self._handle_speed_limit(sumo_connection=sumo_connection)
        self._last_action_time = simulation_time

    def get_observations(self, sumo_connection: traci.Connection, simulation_time: float,
                         vehicles__acc_lane_waiting_times: dict[str, dict[str, float]]) -> np.ndarray:
        # SUMO traffic light state definitions:
        # https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
        _distance_to_tl = self._length
        _traffic_light_state = "g"
        if self._current_leading_vehicle_id is not None:
            _stats = get_next_traffic_light_stats(sumo_connection, vehicle_id=self._current_leading_vehicle_id)
            if _stats is not None:
                _distance_to_tl, _traffic_light_state, _traffic_light_id = _stats
                assert _traffic_light_id == self._associated_traffic_light_id
        traffic_light_state_scalars = traffic_light_state_to_scalars(_traffic_light_state)

        # Lane statistics
        _current_n_vehicles = sumo_connection.lane.getLastStepVehicleNumber(self._lane_id)
        mean_speed = [0] if _current_n_vehicles == 0 else \
            [sumo_connection.lane.getLastStepMeanSpeed(self._lane_id) / self._max_speed]
        traffic_density = [_current_n_vehicles / self._n_vehicles]
        queue = [sumo_connection.lane.getLastStepHaltingNumber(self._lane_id) / self._n_vehicles]
        traffic_light_state_duration = [sumo_connection.trafficlight.getPhaseDuration(self._associated_traffic_light_id) / 30]
        current_speed_limit = [self._current_speed_limit / self._max_speed]
        # TODO Add vehicles waiting time

        observations = traffic_light_state_scalars + mean_speed + traffic_density + queue + \
                       traffic_light_state_duration + current_speed_limit

        # Vehicle-based data
        if self.config.observations_consider_leading_vehicle is True:
            leading_vehicle_speed = [0]
            if self._current_leading_vehicle_id is not None:
                leading_vehicle_speed = [sumo_connection.vehicle.getSpeed(self._current_leading_vehicle_id) / self._max_speed]
            distance_to_traffic_light = [_distance_to_tl / self._length]
            _distance_to_vh = self._current_leading_vehicle_id_distance
            distance_to_next_standing_vehicle = [_distance_to_vh / self._length if _distance_to_vh is not None else 1]
            observations += leading_vehicle_speed + distance_to_traffic_light + distance_to_next_standing_vehicle

        # Some more observations
        if self.config.observations_add_neighboring_lanes is True:
            _nb = self._neighboring_lane_ids
            neighbors_mean_speed = [sumo_connection.lane.getLastStepMeanSpeed(l) / self._max_speed for l in _nb]
            neighbors_density = [sumo_connection.lane.getLastStepVehicleNumber(l) / self._n_vehicles for l in _nb]
            observations += neighbors_mean_speed + neighbors_density

        observations = np.array(observations, dtype=np.float32)
        return observations

    def get_reward(self, sumo_connection: traci.Connection, simulation_time: float,
                   vehicles__acc_lane_waiting_times: dict[str, dict[str, float]],
                   lanes__vehicle_usage_times: dict[str, dict[str, float]]) -> float:
        if self.config.reward_type == "average-speed":
            _speed = sumo_connection.lane.getLastStepMeanSpeed(self._lane_id)
            self._last_reward = _speed / self._max_speed
        elif self.config.reward_type == "diff-waiting-time":
            waiting_time = self._get_waiting_time(sumo_connection=sumo_connection, lane_id=self._lane_id,
                                                  vehicles__lane_waiting_times=vehicles__acc_lane_waiting_times) / 100.0
            self._last_reward = self._last_waiting_time - waiting_time
            self._last_waiting_time = waiting_time
        elif self.config.reward_type == "clipped-waiting-time":
            waiting_time = self._get_waiting_time(sumo_connection=sumo_connection, lane_id=self._lane_id,
                                                  vehicles__lane_waiting_times=vehicles__acc_lane_waiting_times) / 100.0
            self._last_reward = -float(np.clip(waiting_time, -1, 1))
        elif self.config.reward_type == "diff-total-co2-emission":
            total_co2_emission = sumo_connection.lane.getCO2Emission(self._lane_id) / self._n_vehicles
            self._last_reward = (self._last_co2_emission - total_co2_emission) / 1000
            self._last_co2_emission = total_co2_emission
        elif self.config.reward_type == "total-co2-emission":
            total_co2_emission = sumo_connection.lane.getCO2Emission(self._lane_id) / self._n_vehicles
            self._last_reward = -total_co2_emission / 3000
        elif self.config.reward_type == "smoothed-total-co2-emission":
            total_co2_emission = sumo_connection.lane.getCO2Emission(self._lane_id) / self._n_vehicles
            self._co2_emission_memory.append(total_co2_emission)
            smoothed = sum(v for v in self._co2_emission_memory) / len(self._co2_emission_memory)
            self._last_reward = -smoothed / 3000
        elif self.config.reward_type == "time-spent-on-lanes":
            time_spent_on_lane = self._get_time_spent_on_lane(self._lane_id, lanes__vehicle_usage_times)
            self._last_reward = -time_spent_on_lane / 50 / self._n_vehicles
        else:
            raise RuntimeError("We should not have ended-up here!")
        return self._last_reward

    def get_infos(self, sumo_connection: traci.Connection, simulation_time: float) -> dict[str, float]:
        return {}

    @staticmethod
    def _get_mean_co2_emission_per_vehicle(sumo_connection: traci.Connection, lane_id: str) -> float:
        n_vehicles = sumo_connection.lane.getLastStepVehicleNumber(lane_id)
        if n_vehicles == 0:
            return 0
        mean_co2_emission = sumo_connection.lane.getCO2Emission(lane_id) / n_vehicles
        return mean_co2_emission

    @staticmethod
    def _get_waiting_time(sumo_connection: traci.Connection, lane_id: str,
                          vehicles__lane_waiting_times: dict[str, dict[str, float]]) -> float:
        _vehicle_ids = sumo_connection.lane.getLastStepVehicleIDs(lane_id)
        _wait_time = sum([vehicles__lane_waiting_times[v][lane_id] for v in _vehicle_ids])
        return _wait_time

    @staticmethod
    def _get_time_spent_on_lane(lane_id: str, lanes__vehicle_usage_times: dict[str, dict[str, float]]) -> float:
        """Returns the accumulated amount of time *CURRENTLY PRESENT* vehicles spent on this lane, so far."""
        time_spent = sum(lanes__vehicle_usage_times[lane_id].values())
        return time_spent

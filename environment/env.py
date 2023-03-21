from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Tuple, Optional, Union, cast, Any
import traceback

import gym
import numpy as np
import traci.exceptions
from gym.core import ActType, ObsType
from gym.utils.env_checker import check_env
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.pre_checks.env import check_gym_environments, check_multiagent_environments
from ray.rllib.utils.typing import MultiAgentDict

from .actuators import BaseActuator, TrafficLight, Lane, LaneCompound
from .base_env import BaseGymEnv
from .util import episode_end_metrics


class GymEnv(BaseGymEnv):
    reward_range = (-float('inf'), float('inf'))
    metadata = {}

    @dataclass
    class Config(BaseGymEnv.Config):
        single_agent: bool = False  # Acts as single-agent environment. Most likely used for testing purposes
        trafficlight_config: TrafficLight.Config = TrafficLight.Config()
        lane_config: Optional[Lane.Config] = None  # Mutually exclusive w/ lane_compound_config
        lane_compound_config: Optional[LaneCompound.Config] = None  # Mutually exclusive w/ lane_config
        simulation_end_time: float = 20_000  # At this time, the environment returns done == True (i.e. episode ended)
        skip_busy_actuators: bool = True  # Actors which currently don't accept actions will be omitted in observations
        allow_skip_idle_time_steps: bool = True  # Skips time steps if all actors are busy. False is helpful for eval

        @cached_property
        def simulation_duration(self) -> float:
            """Provides the duration of a simulation."""
            return self.simulation_end_time - self.simulation_begin_time

    def __init__(self, config: Config, do_output_info: bool):
        assert config.start_simulator_on_init is True, "This environment was designed to start its simulator at _init_!"
        assert config.lane_config is None or config.lane_compound_config is None, \
            "Lane & lane-compound configs are mutually exclusive. You may either provide one of them, or none at all!"

        super(GymEnv, self).__init__(config=config)
        self.config = cast(self.Config, self.config)
        self._do_output_info = do_output_info

        self._vehicles__acc_lane_waiting_times: dict[str, dict[str, float]] = \
            defaultdict(lambda: defaultdict(float))  # Used to keep track of accumulated vehicle wait times
        self._lanes__vehicle_usage_times: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._scenario_lane_ids = set(l for l in self._sumo_connection.lane.getIDList() if not l.strip().startswith(":"))
        self._skip_actuators_condition = (lambda *x: True) if self.config.skip_busy_actuators is False else \
            (lambda act, conn, time: act.accepts_new_actions(sumo_connection=conn, simulation_time=time))
        _connection = self._sumo_connection
        _simulation_time = _connection.simulation.getTime()

        # Create actuator instances
        _traffic_light_instances = TrafficLight.construct_instances(
            sumo_connection=_connection, simulation_time=_simulation_time,
            actuator_config=self.config.trafficlight_config)
        assert len(_traffic_light_instances) == len(set(t.id for t in _traffic_light_instances))  # No ID duplicates!
        _traffic_light_instances = sorted(_traffic_light_instances, key=lambda x: x.id)  # Sort so we can load models!
        self._traffic_lights: dict[str, TrafficLight] = \
            {f"{tl.__class__.__name__}_{i}[{tl.id}]": tl for i, tl in enumerate(_traffic_light_instances)}

        if self.config.lane_config is not None:
            _lane_instances = Lane.construct_instances(
                sumo_connection=_connection, simulation_time=_simulation_time, actuator_config=self.config.lane_config)
            assert len(_lane_instances) == len(set(t.id for t in _lane_instances))  # No ID duplicates!
            _lane_instances = sorted(_lane_instances, key=lambda x: x.id)  # Sort so we can load saved models!
            self._lanes: dict[str, Lane] = \
                {f"{lane.__class__.__name__}_{i}[{lane.id}]": lane for i, lane in enumerate(_lane_instances)}
        if self.config.lane_compound_config is not None:
            _lc_instances = LaneCompound.construct_instances(
                sumo_connection=_connection, simulation_time=_simulation_time, actuator_config=self.config.lane_compound_config)
            assert len(_lc_instances) == len(set(t.id for t in _lc_instances))  # No ID duplicates!
            _lc_instances = sorted(_lc_instances, key=lambda x: x.id)  # Sort so we can load saved models!
            self._lane_compounds: dict[str, LaneCompound] = \
                {f"{lc.__class__.__name__}_{i}[{lc.id}]": lc for i, lc in enumerate(_lc_instances)}
        # TODO Create others

        self.reset()

        if config.single_agent is True:
            self.observation_space = next(iter(self.actuators.values())).observation_space
            self.action_space = next(iter(self.actuators.values())).action_space
        elif config.single_agent is False:
            self.observation_space = \
                gym.spaces.Dict({name: actuator.observation_space for name, actuator in self.actuators.items()})
            self.action_space = gym.spaces.Dict({name: _act.action_space for name, _act in self.actuators.items()})

    @cached_property
    def actuators(self) -> dict[str, BaseActuator]:
        """Returns all present actuator instances, along with their internal names."""
        if self.config.single_agent is True:
            _key, _val = next(iter(self._traffic_lights.items()))
            return {_key: _val}
        _actuators = {**self._traffic_lights}
        if self.config.lane_config is not None:
            _actuators.update(self._lanes)
        if self.config.lane_compound_config is not None:
            _actuators.update(self._lane_compounds)
        # TODO Add others
        return _actuators

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,
              **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        creation_status = super(GymEnv, self).reset(seed=seed, return_info=return_info, **kwargs)
        assert creation_status in ("new simulation started", "simulation state loaded")

        self._vehicles__acc_lane_waiting_times.clear()
        self._lanes__vehicle_usage_times.clear()
        _connection = self._sumo_connection
        _simulation_time: float = _connection.simulation.getTime()
        assert _simulation_time == self.config.simulation_begin_time

        for _actuator in self.actuators.values():
            _actuator.reset(sumo_connection=_connection, simulation_time=_simulation_time)

        _observations = self._get_observations(simulation_time=_simulation_time)
        assert len(_observations) == len(self.actuators), "Upon reset each actuator is expected to return observations"
        if self.config.single_agent is True:
            _observations = next(iter(_observations.values()))
        if return_info is True:
            _info = self._get_infos(simulation_time=_simulation_time)
            return _observations, _info
        return _observations

    def seed(self, seed=None):
        """This function is implemented despite its deprecation, in order to support RLLib's seeding scheme."""
        # super(RLTrafficGymEnv, self).seed(seed=seed)  -->  Not calling this, because it would produce a warning
        self.reset(seed=seed)  # --> Causes the env to restart SUMO

    def _get_observations(self, simulation_time: float) -> dict[str, np.ndarray]:
        _connection = self._sumo_connection
        _vehicles__acc_lane_waiting_times = self._vehicles__acc_lane_waiting_times
        observations = {name: act.get_observations(sumo_connection=_connection, simulation_time=simulation_time,
                                                   vehicles__acc_lane_waiting_times=_vehicles__acc_lane_waiting_times)
                        for name, act in self.actuators.items()
                        if self._skip_actuators_condition(act, _connection, simulation_time) is True}
        return observations

    def _get_rewards(self, simulation_time: float) -> dict[str, float]:
        _connection = self._sumo_connection
        rewards = {name: act.get_reward(sumo_connection=_connection, simulation_time=simulation_time,
                                        vehicles__acc_lane_waiting_times=self._vehicles__acc_lane_waiting_times,
                                        lanes__vehicle_usage_times=self._lanes__vehicle_usage_times)
                   for name, act in self.actuators.items()
                   if self._skip_actuators_condition(act, _connection, simulation_time) is True}
        return rewards

    def _get_infos(self, simulation_time: float) -> dict[str, Any]:
        if self._do_output_info is False:
            return {}
        _connection = self._sumo_connection
        infos = {name: act.get_infos(self._sumo_connection, simulation_time) for name, act in self.actuators.items()
                 if self._skip_actuators_condition(act, _connection, simulation_time) is True}
        return infos

    def get_intra_episode_metrics(self) -> dict[str, float]:
        """Returns a dictionary that holds most recent metrics."""
        _connection = self._sumo_connection
        _simulation_time = _connection.simulation.getTime()
        metrics = {
            "reward_sum": sum(act.last_reward for act in self.actuators.values() if act.last_reward is not None),
            "simulation_time": _simulation_time,
            "n_vehicles_enqueued": sum(t.get_total_queued(_connection) for t in self._traffic_lights.values()),
            "total_wait_time":
                sum(sum(t._get_waiting_time_per_lane(_connection, self._vehicles__acc_lane_waiting_times).values())
                    for t in self._traffic_lights.values())
        }
        return metrics

    def get_episode_end_metrics(self) -> dict[str, Union[bool, float, int, dict[str, float]]]:
        """
        Returns those metrics that are provided once after each episode. Must be called before calling
        reset() - otherwise, important data get lost.
        """
        _connection = self._sumo_connection
        _sim_time = _connection.simulation.getTime()
        _is_done = _sim_time >= self.config.simulation_end_time or _connection.simulation.getMinExpectedNumber() == 0
        metrics = {
            "simulation_time": _sim_time,
            "done": _is_done  # Denotes if the episode has ended
        }

        if self.config.do_process_trip_info is True:
            assert len(self.trip_info_objects) > 0, \
                "No trip_info objects available. There were either no vehicles in this episode, or reset() " \
                "was called right before calling this method!"
            _resco_delay = episode_end_metrics.get_resco_episode_delay(
                trip_info_objects=self.trip_info_objects, route_file_path=self.config.route_file_path,
                simulation_end_time=self.config.simulation_end_time)
            _resco_delay_no_undeparted_vehicles = episode_end_metrics.get_resco_episode_delay(
                trip_info_objects=self.trip_info_objects, route_file_path=None, simulation_end_time=None)
            metrics.update({
                "n_finished_trips": len(self.trip_info_objects),  # The number of vehicle trips during this episode
                "resco_delay": _resco_delay,
                "resco_delay_no_undeparted_vehicles": _resco_delay_no_undeparted_vehicles,
                "emissions": episode_end_metrics.get_median_emissions(self.trip_info_objects),
            })
        return metrics

    def get_current_lane_speed_limits_norm(self) -> dict[str, float]:
        """
        Returns the current speed limit of all speed-controlled lanes, as values in the range [0; 1].

        @return: A dictionary that maps lane-IDs to their respective normalized current speed limit.
        """
        assert self.config.lane_config is not None or self.config.lane_compound_config is not None
        if self.config.lane_config is not None:
            lane_ids__speed_limits = {lane.id: lane.current_speed_limit_norm for lane in self._lanes.values()}
        elif self.config.lane_compound_config is not None:
            lane_ids__speed_limits = {}
            for _lane_compound in self._lane_compounds.values():
                lane_ids__speed_limits.update(_lane_compound.current_speed_limits_norm)
        else:
            raise RuntimeError("We should not have ended-up here!")
        return lane_ids__speed_limits

    def render(self, mode="human"):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        _actuators = self.actuators
        _connection = self._sumo_connection
        _initial_simulation_time = _simulation_time = _connection.simulation.getTime()

        if self.config.single_agent is True:
            next(iter(self.actuators.values())).act(_connection, simulation_time=_simulation_time, action=action)
        elif self.config.single_agent is False:
            assert isinstance(action, dict)
            assert all(name in _actuators for name in action.keys())
            for _name, _action in action.items():
                _actuators[_name].act(sumo_connection=_connection, simulation_time=_simulation_time, action=_action)

        # Step the environment, until any of the actuators accepts new actions
        while True:
            _is_done = _simulation_time >= self.config.simulation_end_time or \
                       _connection.simulation.getMinExpectedNumber() == 0
            if _is_done is True:
                break
            _connection.simulationStep()
            _simulation_time: float = _connection.simulation.getTime()
            for _act in _actuators.values():
                _act.tick(sumo_connection=_connection, simulation_time=_simulation_time)
            if self.config.skip_busy_actuators is False or self.config.allow_skip_idle_time_steps is False:
                break
            if any(_act.accepts_new_actions(_connection, _simulation_time) for _act in _actuators.values()):
                break

        # Keep track of a few timings. These values are crucial for reward & obs computation of actuators
        self._track_vehicle_waiting_times()
        self._track_time_spent_on_lanes(sim_time_step_length=_simulation_time - _initial_simulation_time)

        # Gather observations, rewards, etc.
        _observations = self._get_observations(simulation_time=_simulation_time)
        _rewards = self._get_rewards(simulation_time=_simulation_time)
        _infos = self._get_infos(simulation_time=_simulation_time)

        _dones = {name: False for name in _actuators.keys()}
        _dones["__all__"] = _is_done

        if self.config.single_agent is True:
            _obs, _rew, _info = \
                next(iter(_observations.values())) if len(_observations) else None, \
                next(iter(_rewards.values())) if len(_rewards) else None, \
                next(iter(_infos.values())) if len(_infos) else None
            return _obs, _rew, _dones["__all__"], _info
        return _observations, _rewards, _dones, _infos

    def _track_vehicle_waiting_times(self) -> None:
        """Keeps track of each vehicle's per-lane waiting time. Called once per environment step."""
        _connection = self._sumo_connection
        for vehicle_id in _connection.vehicle.getIDList():
            if _connection.vehicle.getWaitingTime(vehicle_id) == 0:
                continue
            _lane_id = _connection.vehicle.getLaneID(vehicle_id)
            _acc_waiting_time = _connection.vehicle.getAccumulatedWaitingTime(vehicle_id)
            _other_lanes_waiting_time = sum([self._vehicles__acc_lane_waiting_times[vehicle_id][lane]
                                             for lane in self._vehicles__acc_lane_waiting_times[vehicle_id].keys()
                                             if lane != _lane_id])
            self._vehicles__acc_lane_waiting_times[vehicle_id][_lane_id] = _acc_waiting_time - _other_lanes_waiting_time

    def _track_time_spent_on_lanes(self, sim_time_step_length: float) -> None:
        """
        Measures the accumulated time that vehicles use a lane. Once they move on to another lane,
        their lane-time resets to zero.
        """
        _connection = self._sumo_connection
        # Determine which vehicles currently use which lanes
        _lanes__vehicles: dict[str, list[str]] = defaultdict(list)
        for _vehicle_id in _connection.vehicle.getIDList():
            _lane_id = _connection.vehicle.getLaneID(_vehicle_id)
            _lanes__vehicles[_lane_id] += [_vehicle_id]
        # Take care of how long vehicles spend time on each lane
        for _lane_id in self._scenario_lane_ids:
            _vehicles_on_lane = set(_lanes__vehicles[_lane_id])
            for _vehicle_id in _vehicles_on_lane:
                self._lanes__vehicle_usage_times[_lane_id][_vehicle_id] += sim_time_step_length
            _vehicles_not_on_lane = set(self._lanes__vehicle_usage_times[_lane_id].keys()) - _vehicles_on_lane
            for _vehicle_id in _vehicles_not_on_lane:
                self._lanes__vehicle_usage_times[_lane_id].pop(_vehicle_id, None)


class RLLibEnv(MultiAgentEnv):
    """Wraps the GymEnv, such that it can be utilized with RLLib."""
    def __init__(self, env: gym.Env):
        super(RLLibEnv, self).__init__()
        _unwrapped_env = env.unwrapped
        assert isinstance(_unwrapped_env, GymEnv)
        assert _unwrapped_env.config.single_agent is False
        self.env = env
        self.config = _unwrapped_env.config
        env.reset()

        # The following attributes are read by MultiAgentEnv
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._agent_ids = set(_unwrapped_env.actuators.keys())
        self._spaces_in_preferred_format = True

    @property
    def unwrapped(self) -> gym.Env:
        return self.env.unwrapped

    def reset(self) -> MultiAgentDict:
        return self.env.reset()

    def step(self, action_dict: MultiAgentDict) \
            -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        _observations, _rewards, _dones, _infos = self.env.step(action=action_dict)
        return _observations, _rewards, _dones, _infos  # noqa

    def render(self, mode=None) -> None:
        pass

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def close(self):
        self.env.close()


def _env_config_provider(single_agent: bool, use_gui: bool = True):
    file_path = Path(__file__).parent
    config = GymEnv.Config(use_gui=use_gui,
                           net_file_path=file_path/"unit_test_data/single-intersection.net.xml",
                           route_file_path=file_path/"unit_test_data/single-intersection.rou.xml",
                           single_agent=single_agent)
    return config


def test_gym_env__dev():
    """This function serves development purposes and might not be properly "cleaned-up"."""
    from environment import env_config_templates
    _env_config = env_config_templates.resco_grid4x4()
    _env_config.use_gui = True
    _env_config.trafficlight_config.reward_type = "time-spent-on-lanes"
    # _env_config.lane_config = Lane.Config(physical_prediction_measure="speed-change", reward_type="time-spent-on-lanes")
    _env_config.lane_compound_config = LaneCompound.Config(cluster_lanes_by="traffic-light", physical_prediction_measure="speed-change", reward_type="time-spent-on-lanes")
    # _env_config.trafficlight_config.observations_add_leading_vehicle = True
    # _env_config.trafficlight_config.enhance_short_lanes = True
    env = GymEnv(config=_env_config, do_output_info=False)
    env.reset()
    env.get_intra_episode_metrics()
    for _ in range(100):
        env.step(env.action_space.sample())
    env.get_episode_end_metrics()
    env.reset()
    env.get_intra_episode_metrics()
    env.close()


def test_gym_env__check_env_1():
    config = _env_config_provider(single_agent=True, use_gui=True)
    env = GymEnv(config=config, do_output_info=False)
    check_env(env)  # Checker comes from Gym


def test_gym_env__check_env_2():
    config = _env_config_provider(single_agent=True, use_gui=True)
    env = GymEnv(config=config, do_output_info=False)
    check_gym_environments(env)  # Checker comes from RLLib


def test_rllib_env__check_env():
    gym_env = GymEnv(config=_env_config_provider(single_agent=False, use_gui=True), do_output_info=False)
    rllib_env = RLLibEnv(env=gym_env)
    check_multiagent_environments(rllib_env)


def test_rllib_env__dev():
    gym_env = GymEnv(config=_env_config_provider(single_agent=False, use_gui=True), do_output_info=False)
    rllib_env = RLLibEnv(env=gym_env)
    rllib_env.reset()
    for _ in range(100):
        rllib_env.step(rllib_env.action_space.sample())
    rllib_env.close()


def test_rllib_env__runtime():
    from .env_config_templates import resco_grid4x4
    env_config = resco_grid4x4()
    gym_env = GymEnv(config=env_config, do_output_info=False)
    rllib_env = RLLibEnv(env=gym_env)
    rllib_env.reset()

    started_at = datetime.now()
    for _ in range(200):
        rllib_env.step(rllib_env.action_space.sample())
    rllib_env.close()
    print(f"\n\nThe process took {(datetime.now()-started_at).total_seconds():.1f}s")

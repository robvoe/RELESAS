from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import gym
import numpy as np
import traci

from util.base_config import BaseConfig


class BaseActuator(ABC):
    @dataclass
    class Config(BaseConfig):
        pass

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    @abstractmethod
    def construct_instances(sumo_connection: traci.Connection, simulation_time: float, actuator_config: Config,
                            **kwargs) -> List["BaseActuator"]:
        """
        This method provides all necessary actuator instances for the scenario at hand. This method will be invoked
        from env.py during its construction phase.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Provides the unique SUMO-related id of this actuator."""
        pass

    @property
    @abstractmethod
    def last_reward(self) -> Optional[float]:
        """Provides the reward that was previously computed. If none was computed so far, the result is None."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @abstractmethod
    def reset(self, sumo_connection: traci.Connection, simulation_time: float) -> None:
        """
        Called upon env reset. Re-initializes the internal states. Note that right after resetting, an actuator has to
        accept new actions.
        """
        pass

    @abstractmethod
    def tick(self, sumo_connection: traci.Connection, simulation_time: float) -> None:
        """Called whenever the simulation stepped forward."""
        pass

    @abstractmethod
    def accepts_new_actions(self, sumo_connection: traci.Connection, simulation_time: float) -> bool:
        """Returns whether the actuator accepts new actions."""
        pass

    @abstractmethod
    def act(self, sumo_connection: traci.Connection, simulation_time: float, action: Union[int, float, np.ndarray]):
        pass

    @abstractmethod
    def get_observations(self, sumo_connection: traci.Connection, simulation_time: float,
                         vehicles__acc_lane_waiting_times: dict[str, dict[str, float]]) -> np.ndarray:
        """
        Gets called by the env in order to obtain current state/observations.

        @param sumo_connection: Connection to SUMO
        @param simulation_time: Current simulation time
        @param vehicles__acc_lane_waiting_times: Dictionary of accumulated waiting times per vehicle and lane.

        @return: The actuator's current observations
        """
        pass

    @abstractmethod
    def get_reward(self, sumo_connection: traci.Connection, simulation_time: float,
                   vehicles__acc_lane_waiting_times: dict[str, dict[str, float]],
                   lanes__vehicle_usage_times: dict[str, dict[str, float]]) -> float:
        """
        Is called by the env in order to provide the current reward.

        @param sumo_connection: Connection to SUMO
        @param simulation_time: Current simulation time
        @param vehicles__acc_lane_waiting_times: Dictionary of accumulated waiting times per vehicle and lane.
        @param lanes__vehicle_usage_times: Dictionary that maps lane IDs to vehicle-specific lane usage times.

        @return: The actuator's current reward
        """
        pass

    @abstractmethod
    def get_infos(self, sumo_connection: traci.Connection, simulation_time: float) -> dict[str, float]:
        """Returns performance metrics that will be outputted in the info field of the environment"""
        pass

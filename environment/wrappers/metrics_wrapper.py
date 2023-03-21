import pickle
import time
from pathlib import Path
from typing import Union, Tuple, cast, Optional

import numpy as np
import gym
import pandas as pd
import yaml
from gym.core import ObsType, ActType

from environment.env import GymEnv


class MetricsWrapper(gym.Wrapper):
    """A wrapper that automatically logs metrics to CSV/YAML files."""
    def __init__(self, env: gym.Env, intra_episode_csv_file_path: Optional[Path],
                 episode_end_yaml_file_path: Optional[Path], episode_end_trip_info_objects_pkl_path: Optional[Path],
                 intra_episode_logging_frequency: int = 1):
        """
        Constructor

        @param env: Environment to be wrapped
        @param intra_episode_csv_file_path: Name of the resulting CSV files which store intra-episode metrics. Data
                                            from within first episode will be saved to a file with suffix .1, data
                                            from second episode are saved to a file with suffix .2, etc. If None,
                                            no intra-episode data will be stored.
        @param intra_episode_logging_frequency: Defines, at which Nth step we will log intra-episode metrics. The most
                                                complete logging (i.e. every step) is done at frequency == 1. Be aware
                                                the denser the logging steps, the larger the calculation overhead!
        @param episode_end_yaml_file_path: YAML files where episode-end metrics are stored. Data from within first
                                           episode will be saved to a file with suffix .1, data from second episode
                                           are saved to a file with suffix .2, etc. If None, no episode-end metrics
                                           will be stored.
        @param episode_end_trip_info_objects_pkl_path: Resulting pickle (pkl) files where episode-end trip-info objects
                                                       are stored. Data from within first episode will be saved to a
                                                       file with suffix .1, data from second episode are saved to a file
                                                       with suffix .2, etc. If None, no trip info objects will be stored
        """
        super(MetricsWrapper, self).__init__(env=env)
        assert intra_episode_logging_frequency >= 1
        assert sum(p is not None for p in (intra_episode_csv_file_path, episode_end_yaml_file_path,
                                           episode_end_trip_info_objects_pkl_path)) >= 1, \
            "At least one of the parameters 'intra_episode_csv_file_path'/'episode_end_yaml_file_path'/" \
            "'episode_end_trip_info_objects_pkl_path' must be provided"

        assert isinstance(env.unwrapped, GymEnv), f"The provided env is no (or does not wrap a) {GymEnv.__name__}!"
        self._unwrapped_env: GymEnv = cast(GymEnv, env.unwrapped)
        self._episode_counter = 1
        self._intra_episode_memory: list[dict[str, float]] = []

        if episode_end_trip_info_objects_pkl_path is not None:
            assert self._unwrapped_env.config.do_process_trip_info is True

        def _check_file_path(_file_path: Path):
            if _file_path is None:
                return
            assert _file_path.parent.is_dir(), f"Destination dir '{_file_path}' does not exist!"
            _generated = self._generate_filename(_file_path)
            assert _generated.exists() is False, f"YAML/CSV file {self._generated} already exists!"
            open(_generated, mode="w", encoding="utf-8").close()  # Create an empty file to check write permissions
            _generated.unlink()

        _check_file_path(intra_episode_csv_file_path)
        _check_file_path(episode_end_yaml_file_path)
        _check_file_path(episode_end_trip_info_objects_pkl_path)
        self._intra_episode_csv_file_path = intra_episode_csv_file_path
        self._episode_end_yaml_file_path = episode_end_yaml_file_path
        self._episode_end_trip_info_objects_pkl_path = episode_end_trip_info_objects_pkl_path

        self._intra_episode_logging_frequency = intra_episode_logging_frequency
        self._env_step = 0
        self._do_log_reward = len(self._unwrapped_env.actuators) == 1

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        _observations, _rewards, _dones, _infos = super().step(action)
        _is_done = _dones is True or (isinstance(_dones, dict) and _dones["__all__"] is True)

        if self._intra_episode_csv_file_path is not None and \
                (self._env_step % self._intra_episode_logging_frequency == 0 or _is_done):
            metrics = self._unwrapped_env.get_intra_episode_metrics()
            metrics["env_step"] = self._env_step
            if self._do_log_reward:
                if isinstance(_rewards, dict) and len(_rewards):
                    metrics["reward"] = float(next(iter(_rewards.values())))
                elif isinstance(_rewards, (float, int, np.float)):
                    metrics["reward"] = float(_rewards)
                else:
                    metrics["reward"] = None
            self._intra_episode_memory.append(metrics)

        self._env_step += 1
        return _observations, _rewards, _dones, _infos

    def _generate_filename(self, original_file_path: Path) -> Path:
        assert self._episode_counter >= 1
        filename = original_file_path.parent / f"{original_file_path.name}.{self._episode_counter}"
        return filename

    def _write_results(self):
        # Write intra-episode metrics
        if self._intra_episode_csv_file_path is not None:
            _csv_filename = self._generate_filename(self._intra_episode_csv_file_path)
            df = pd.DataFrame(self._intra_episode_memory)
            df.to_csv(_csv_filename, index=False, encoding="utf-8", mode="w", sep=",")

        # Write episode-end metrics
        if self._episode_end_yaml_file_path is not None:
            _yaml_filename = self._generate_filename(self._episode_end_yaml_file_path)
            _episode_end_metrics = self._unwrapped_env.get_episode_end_metrics()
            _episode_end_metrics["env_step"] = self._env_step
            _episode_end_metrics["timestamp"] = time.time()
            with open(_yaml_filename, mode="w", encoding="utf-8") as file:
                yaml.dump(_episode_end_metrics, stream=file, sort_keys=False)

        # Write episode-end trip-info objects
        if self._episode_end_trip_info_objects_pkl_path is not None:
            _pkl_filename = self._generate_filename(self._episode_end_trip_info_objects_pkl_path)
            with open(_pkl_filename, mode="wb") as file:
                pickle.dump(self._unwrapped_env.trip_info_objects, file)

    def reset(self, **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        if self._env_step > 1:
            self._write_results()
            self._episode_counter += 1
        self._env_step, self._intra_episode_memory = 0, []
        return super().reset(**kwargs)

    def close(self):
        if self._env_step > 1:
            self._write_results()
        super().close()

    def __del__(self):
        if self._env_step > 1:
            self._write_results()

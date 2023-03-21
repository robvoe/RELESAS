from collections import deque
from pathlib import Path
from typing import cast, Tuple, Optional, Union

import gym
import numpy as np
from gym.core import ActType, ObsType

from environment.env import GymEnv


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, n_frames: int):
        """Stack k last frames."""
        super(FrameStack, self).__init__(env=env)
        assert isinstance(env.unwrapped, GymEnv), f"The provided env is no (or does not wrap a) {GymEnv.__name__}!"
        assert n_frames >= 1
        self._unwrapped_env: GymEnv = cast(GymEnv, env.unwrapped)
        self._n_frames = n_frames

        _obs_spaces_dict = env.observation_space
        assert isinstance(_obs_spaces_dict, gym.spaces.Dict), "The env's observation space has to be a dict space!"
        spaces = {}
        self._stacked_observations = {}
        for _agent_id, _space in _obs_spaces_dict.items():
            _shape, _low, _high = _space.shape, _space.low.min(), _space.high.max()
            _new_space = gym.spaces.Box(low=_low, high=_high, shape=(n_frames, *_shape), dtype=_space.dtype)
            spaces[_agent_id] = _new_space
            self._stacked_observations[_agent_id] = np.zeros_like(_new_space.sample())
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self):
        obs_dict = self.env.reset()
        assert isinstance(obs_dict, dict)
        for _stacked_obs in self._stacked_observations.values():
            _stacked_obs[:] = 0
        for _agent_id, _obs in obs_dict.items():
            self._stacked_observations[_agent_id][-1] = _obs
        return {_id: self._stacked_observations[_id] for _id in obs_dict.keys()}

    def step(self, action):
        obs_dict, reward, dones, info = self.env.step(action)
        assert isinstance(obs_dict, dict)
        for _agent_id, _obs in obs_dict.items():
            _rolled_obs = np.roll(self._stacked_observations[_agent_id], shift=-1, axis=0)
            _rolled_obs[-1] = _obs
            self._stacked_observations[_agent_id] = _rolled_obs
        return {_id: self._stacked_observations[_id] for _id in obs_dict.keys()}, reward, dones, info


class _DummyEnv(GymEnv):
    def __init__(self):
        _space = gym.spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({"agent1": _space, "agent2": _space})
        self.observations = [
            {"agent1": 0, "agent2": 10},
            {"agent1": 1, "agent2": 11},
            {"agent1": 2},
            {"agent1": 3},
            {},
            {"agent1": 4, "agent2": 12},
            {"agent1": 5, "agent2": 13},
            {"agent1": 6, "agent2": 14},
        ]
        self.observations_idx = 0

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,
              **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        self.observations_idx = 0
        return self.observations[self.observations_idx]

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.observations_idx += 1
        assert self.observations_idx < len(self.observations), "Stepping into unknown realms ;-)"
        obs = self.observations[self.observations_idx]
        return obs, 0, False, {}


def test_frame_stack():
    env = _DummyEnv()
    env = FrameStack(env, n_frames=3)

    def _test(_obs, _agent1, _agent2):
        if _agent1 is None: assert "agent1" not in _obs
        else: np.testing.assert_allclose(obs["agent1"], _agent1)
        if _agent2 is None: assert "agent2" not in _obs
        else: np.testing.assert_allclose(obs["agent2"], _agent2)

    obs = env.reset()
    _test(obs, _agent1=np.array([[0], [0], [0]]), _agent2=np.array([[0], [0], [10]]))

    obs, _, _, _ = env.step(action=None)
    _test(obs, _agent1=np.array([[0], [0], [1]]), _agent2=np.array([[0], [10], [11]]))

    obs, _, _, _ = env.step(action=None)
    _test(obs, _agent1=np.array([[0], [1], [2]]), _agent2=None)

    obs, _, _, _ = env.step(action=None)
    _test(obs, _agent1=np.array([[1], [2], [3]]), _agent2=None)

    obs, _, _, _ = env.step(action=None)
    _test(obs, _agent1=None, _agent2=None)

    obs, _, _, _ = env.step(action=None)
    _test(obs, _agent1=np.array([[2], [3], [4]]), _agent2=np.array([[10], [11], [12]]))

    obs, _, _, _ = env.step(action=None)
    _test(obs, _agent1=np.array([[3], [4], [5]]), _agent2=np.array([[11], [12], [13]]))

    obs, _, _, _ = env.step(action=None)
    _test(obs, _agent1=np.array([[4], [5], [6]]), _agent2=np.array([[12], [13], [14]]))

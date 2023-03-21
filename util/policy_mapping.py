from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict, Callable
import os.path

import gym.spaces
import numpy as np
from ray.rllib.policy.policy import PolicySpec


@dataclass
class AgentSpaces:
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space

    @property
    def observation_space_str(self):
        return str(self.observation_space)

    @property
    def action_space_str(self):
        return str(self.action_space)


def get_generic_policy_mapping(agent_spaces: Dict[str, AgentSpaces], weights_sharing: bool) \
        -> Tuple[Dict[str, PolicySpec], Callable, Dict]:
    """
    Utility function that groups agents with identical observation/action spaces, and provides both
    - policy specs
    - a policy mapping function
    which then can be passed to RLLib right away.
    """
    policy_specs: Dict[str, PolicySpec] = {}
    policy_mapping: Dict[str, str] = {}  # Maps from agent ids to policy ids
    if weights_sharing is True:
        _distinct_observation_spaces = set(a.observation_space_str for a in agent_spaces.values())
        _distinct_action_spaces = set(a.action_space_str for a in agent_spaces.values())
        _prefix_counters = defaultdict(int)
        for _obs_str in _distinct_observation_spaces:
            _obs_agents = {k: v for k, v in agent_spaces.items() if v.observation_space_str == _obs_str}
            for _act_str in _distinct_action_spaces:
                _act_agents = {k: v for k, v in _obs_agents.items() if v.action_space_str == _act_str}
                if len(_act_agents) == 0:
                    continue
                _common_prefix = os.path.commonprefix(tuple(_act_agents.keys()))
                _common_prefix = _common_prefix.strip("0123456789_").lower()
                _common_prefix = "policy" if len(_common_prefix) == 0 else _common_prefix + "_policy"
                _policy_name = f"{_common_prefix}_{_prefix_counters[_common_prefix]}"
                _prefix_counters[_common_prefix] += 1
                _spaces: AgentSpaces = next(iter(_act_agents.values()))
                policy_specs[_policy_name] = PolicySpec(policy_class=None, observation_space=_spaces.observation_space,
                                                        action_space=_spaces.action_space, config=None)
                policy_mapping.update({k: _policy_name for k in _act_agents.keys()})
    else:
        for _agent_name, _spaces in agent_spaces.items():
            _policy_name = _agent_name.lower() + "_policy"
            policy_specs[_policy_name] = PolicySpec(policy_class=None, observation_space=_spaces.observation_space,
                                                    action_space=_spaces.action_space, config=None)
            policy_mapping[_agent_name] = _policy_name

    def _policy_mapping_fn(_agent_id, *_args, **_kwargs):
        assert _agent_id in policy_mapping
        return policy_mapping[_agent_id]
    return policy_specs, _policy_mapping_fn, policy_mapping


def test_get_policy_mapping__no_weights_sharing():
    _any_space = gym.spaces.Discrete(2)
    _agent_spaces = {
        "TrafficLight_0": AgentSpaces(observation_space=_any_space, action_space=_any_space),
        "TrafficLight_1": AgentSpaces(observation_space=_any_space, action_space=_any_space),
        "TrafficLight_2": AgentSpaces(observation_space=_any_space, action_space=_any_space),
    }
    policy_specs, policy_mapping_fn, _ = get_generic_policy_mapping(_agent_spaces, weights_sharing=False)

    assert policy_mapping_fn("TrafficLight_0") == "trafficlight_0_policy"
    assert policy_mapping_fn("TrafficLight_1") == "trafficlight_1_policy"
    assert policy_mapping_fn("TrafficLight_2") == "trafficlight_2_policy"


def test_get_policy_mapping__weights_sharing():
    _small_obs_vector = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    _small_obs_space = gym.spaces.Box(low=np.zeros_like(_small_obs_vector), high=np.ones_like(_small_obs_vector))
    _large_obs_vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    _large_obs_space = gym.spaces.Box(low=np.zeros_like(_large_obs_vector), high=np.ones_like(_large_obs_vector))

    _small_act_space = gym.spaces.Discrete(2)
    _large_act_space = gym.spaces.Discrete(5)

    _agent_spaces = {
        # First group of identical obs/act spaces
        "Agent_0": AgentSpaces(observation_space=_small_obs_space, action_space=_small_act_space),
        "Agent_1": AgentSpaces(observation_space=_small_obs_space, action_space=_small_act_space),
        # Second group of identical abs/act spaces
        "Agent_2": AgentSpaces(observation_space=_small_obs_space, action_space=_large_act_space),
        "Agenda_0": AgentSpaces(observation_space=_small_obs_space, action_space=_large_act_space),
        "B2": AgentSpaces(observation_space=_small_obs_space, action_space=_large_act_space),
        # Third group of identical abs/act spaces
        "TrafficLight_0": AgentSpaces(observation_space=_large_obs_space, action_space=_large_act_space),
        "TrafficLight_1": AgentSpaces(observation_space=_large_obs_space, action_space=_large_act_space),
        "TrafficLight_2": AgentSpaces(observation_space=_large_obs_space, action_space=_large_act_space),
    }
    policy_specs, policy_mapping_fn, _ = get_generic_policy_mapping(_agent_spaces, weights_sharing=True)
    assert len(policy_specs) == 3

    first_group_policy_name = [k for k, v in policy_specs.items() if v.action_space == _small_act_space and v.observation_space == _small_obs_space][0]
    assert policy_mapping_fn("Agent_0") == first_group_policy_name
    assert policy_mapping_fn("Agent_1") == first_group_policy_name

    second_group_policy_name = [k for k, v in policy_specs.items() if v.action_space == _large_act_space and v.observation_space == _small_obs_space][0]
    assert policy_mapping_fn("Agent_2") == second_group_policy_name
    assert policy_mapping_fn("Agenda_0") == second_group_policy_name
    assert policy_mapping_fn("B2") == second_group_policy_name

    third_group_policy_name = [k for k, v in policy_specs.items() if v.action_space == _large_act_space and v.observation_space == _large_obs_space][0]
    assert policy_mapping_fn("TrafficLight_0") == third_group_policy_name
    assert policy_mapping_fn("TrafficLight_1") == third_group_policy_name
    assert policy_mapping_fn("TrafficLight_2") == third_group_policy_name

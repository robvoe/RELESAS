"""Contains logic to load & evaluate trained model checkpoints."""
import pickle
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Optional, Union, Callable, cast
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune import register_env

from environment.env import RLLibEnv, GymEnv
from environment.wrappers.frame_stack import FrameStack
from environment.env_config_templates import FUNCTION_NAMES__TEMPLATE_FUNCTIONS
from util.finished_trainings.path_lookup import get_single_trial_dirs
from util.nested_dicts import flatten_nested_dict

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)$", flags=re.IGNORECASE)

_TEMPLATE_SCENARIO_CONFIGS: list[GymEnv.Config] = [fn() for fn in FUNCTION_NAMES__TEMPLATE_FUNCTIONS.values()]

CustomCallback = Callable[[MultiAgentDict, MultiAgentDict], None]

FILENAME__EVALUATION_RUN_RESULTS_TABLE = "evaluation-run-results-table.tsv"
FILENAME__EVALUATION_RESULTS_SUMMARY = "evaluation-results-summary.yaml"


@contextmanager
def load_checkpoint(trial_dir: Path, env_use_gui: bool = None, env_allow_skip_idle_time_steps: Optional[bool] = False,
                    env_skip_busy_actuators: Optional[bool] = True) -> Tuple[Algorithm, RLLibEnv, dict[str, str], str]:
    """
    Loads a trained (checkpointed) model from a trial directory. Additionally, provides a respective environment
    and necessary agent-to-policy mapping information.

    To make sure all envs, handles, etc. are closed after use, this function is implemented as context manager.

    @param trial_dir: Directory of the trial. Must contain a "checkpoints" folder, which we load the checkpoint from.
    @param env_use_gui: Denotes whether to override the "use_gui" option of the returned environment, and which value.
    @param env_allow_skip_idle_time_steps: If not None, overrides the respective entry in EnvConfig. If True, the env
                                           may skip entire time steps if all actors are busy. Skipping might cause
                                           undesired time gaps when collecting evaluation data, so you likely want to
                                           choose False!
    @param env_skip_busy_actuators: If not None, overrides the respective entry in EnvConfig. If True, the env
                                    respects the business of actors and won't force them to accept new actions. If
                                    False, the env forces also busy actors to accept & perform actions every time step.
    @return: Returns the loaded checkpoint, an instance of a fitting environment, as well as agent-to-policy mappings.
    """
    assert trial_dir.is_dir(), f"Given trial dir '{trial_dir}' either not exists or is no folder."
    _checkpoints_dir = trial_dir / "checkpoints"
    assert _checkpoints_dir.is_dir(), f"Given trial dir '{trial_dir}' doesn't contain a {_checkpoints_dir.name} folder!"

    # Search for checkpoints.   Use the one with the highest index, as it holds the best weights
    _checkpoint_indices__paths: dict[int, Path] = {}
    for _p in _checkpoints_dir.glob("**"):
        _match = _CHECKPOINT_PATTERN.match(_p.name)
        if _match is None:
            continue
        _checkpoint_index = int(_match.group(1))
        assert _checkpoint_index not in _checkpoint_indices__paths
        _checkpoint_indices__paths[_checkpoint_index] = _p
    assert len(_checkpoint_indices__paths) > 0, \
        f"The {_checkpoints_dir.name} folder of trial '{trial_dir}' does not hold any checkpoints!"
    _checkpoint_subdir = _checkpoint_indices__paths[max(_checkpoint_indices__paths.keys())]

    # Load env config and build the env
    _env_config_path = trial_dir / "env-config.yaml"
    assert _env_config_path.is_file(), f"Given trial dir '{trial_dir}' contains no '{_env_config_path.name}'!"
    _env_config = GymEnv.Config.load_from_yaml(_env_config_path)

    # If necessary, patch the scenario paths (e.g. if the model was trained on another machine)
    if not _env_config.net_file_path.is_file() or not _env_config.route_file_path.is_file():
        # --> Try to infer correct path to the given scenario name. But this only works, if a template scenario was used
        assert any(_env_config.net_file_stem == cfg.net_file_stem for cfg in _TEMPLATE_SCENARIO_CONFIGS), \
            f"'{_env_config.net_file_stem}' couldn't be found in template scenarios, patching paths impossible!"
        for _cfg in _TEMPLATE_SCENARIO_CONFIGS:
            if _env_config.net_file_path.name == _cfg.net_file_path.name and \
                    _env_config.route_file_path.name == _cfg.route_file_path.name:
                _env_config.net_file_path = _cfg.net_file_path
                _env_config.route_file_path = _cfg.route_file_path
                break
        assert _env_config.net_file_path.is_file() and _env_config.route_file_path.is_file(), \
            f"Though the scenario '{_env_config.net_file_stem}' seems to be known, pathing paths was impossible! Weird!"

    # Build the algorithm & load the checkpoint
    _algorithm_params_path = trial_dir / "algorithm-construction-params.pkl"
    assert _algorithm_params_path.is_file(), f"Given trial dir '{trial_dir}' contains no {_algorithm_params_path.name}!"
    with open(_algorithm_params_path, mode="rb") as _file:
        _algorithm_construction_params_dict = pickle.load(_file)
    _algorithm_type = _algorithm_construction_params_dict["algorithm_type"]
    _algorithm_config = _algorithm_construction_params_dict["algorithm_config"]
    policy_mapping_dict: dict[str, str] = _algorithm_construction_params_dict["policy_mapping_dict"]
    _n_stacked_frames: int = _algorithm_construction_params_dict["n_stacked_frames"]

    def _env_provider(*_):
        _env = GymEnv(_env_config, do_output_info=False)
        if _n_stacked_frames >= 2:
            _env = FrameStack(_env, n_frames=_n_stacked_frames)
        # TODO Add env wrappers, if present
        return RLLibEnv(_env)
    register_env("_just_some_name_", env_creator=_env_provider)

    def _policy_mapping_fn(_agent_id, *_args, **_kwargs): return policy_mapping_dict[_agent_id]
    _algorithm_config["multiagent"]["policy_mapping_fn"] = _policy_mapping_fn
    _algorithm_config["num_workers"] = 0
    _algorithm_config["num_envs_per_worker"] = 1

    if ray.is_initialized() is False:
        ray.init(local_mode=True)

    _algorithm: Algorithm = _algorithm_type(env="_just_some_name_", config=_algorithm_config)
    _algorithm.restore(checkpoint_path=str(_checkpoint_subdir))

    if env_use_gui is not None:
        _env_config.use_gui = env_use_gui
    if env_allow_skip_idle_time_steps is not None:
        _env_config.allow_skip_idle_time_steps = env_allow_skip_idle_time_steps
    if env_skip_busy_actuators is not None:
        _env_config.skip_busy_actuators = env_skip_busy_actuators
    _env = _env_provider()
    yield _algorithm, _env, policy_mapping_dict, _env_config.net_file_stem

    # Destroy all envs after use
    _algorithm.cleanup()
    _env.close()


def run_single_inference(algorithm: Algorithm, rllib_env: RLLibEnv, policy_mapping_dict: dict[str, str],
                         custom_callback: Optional[CustomCallback]) \
        -> dict[str, Union[bool, float, int, dict[str, float]]]:
    """
    Runs a single inference episode on a given environment, with a given (trained) algorithm. This method is usually
    called with outputs from "load_checkpoint()" method.

    @param algorithm: The trained algorithm we want to use for inference.
    @param rllib_env: The environment we want to use for inference.
    @param policy_mapping_dict: A dictionary that maps from AgentIDs to PolicyIDs.
    @param custom_callback: An optional callback that's called right before stepping the environment. It is supplied
                            with the most recent observations and the respective actions. This callback can be
                            utilized by the user to collect environment statistics.
    @return: Returns an episode-end metrics dictionary.
    """
    _obs_dict = rllib_env.reset()
    _dones = {"__all__": False}
    while _dones["__all__"] is False:
        _actions_dict = {}
        for _agent_id, _o in _obs_dict.items():
            _agent_action = algorithm.compute_single_action(_o, policy_id=policy_mapping_dict[_agent_id])
            _actions_dict[_agent_id] = _agent_action

        if custom_callback is not None:
            custom_callback(_obs_dict, _actions_dict)
        _obs_dict, _, _dones, _ = rllib_env.step(_actions_dict)

    _gym_env = cast(GymEnv, rllib_env.unwrapped)

    return _gym_env.get_episode_end_metrics()


def _parallel_worker_main(trial_dir: Path, env_skip_busy_actuators: Optional[bool]) \
        -> dict[str, Union[bool, float, int, dict[str, float]]]:
    with load_checkpoint(trial_dir, env_use_gui=False, env_allow_skip_idle_time_steps=True,
                         env_skip_busy_actuators=env_skip_busy_actuators) as (algorithm, rllib_env, policy_mapping_dict, net_file_stem):
        episode_end_metrics = run_single_inference(
            algorithm=algorithm, rllib_env=rllib_env, policy_mapping_dict=policy_mapping_dict, custom_callback=None)
    return episode_end_metrics


def run_multiple_inferences(trial_dir: Path, n_runs: int = 20, env_skip_busy_actuators: Optional[bool] = False) \
        -> pd.DataFrame:
    """
    Does exactly the same as "run_single_inference", but runs multiple inferences parallel. The use of custom callbacks
    is -in contrast to the single run variant- not intended.

    @param trial_dir: Directory of the trial. Must contain a "checkpoints" folder, which we load the checkpoint from.
    @param env_skip_busy_actuators: If not None, overrides the respective entry in EnvConfig. If True, the env
                                    respects the business of actors and won't force them to accept new actions. If
                                    False, the env forces also busy actors to accept & perform actions every time step.
    @param n_runs: Number of parallel evaluation runs
    @return: Returns a dataframe that contains the collected episode-end metrics.
    """
    _max_n_workers = int(2/3 * len(os.sched_getaffinity(0)))
    _n_workers = int(np.clip(n_runs, a_min=1, a_max=_max_n_workers))
    with ProcessPoolExecutor(max_workers=_n_workers) as executor:
        _futures = [executor.submit(_parallel_worker_main, trial_dir=trial_dir,
                                    env_skip_busy_actuators=env_skip_busy_actuators) for _ in range(n_runs)]
        _episode_end_metrics_list = [f.result() for f in _futures]

    # --> Flatten nested dicts, if necessary (e.g. emissions-subdict)
    _episode_end_metrics_list = [flatten_nested_dict(e) for e in _episode_end_metrics_list]

    df_aggregated = pd.DataFrame(_episode_end_metrics_list)
    return df_aggregated


def test_single_inference():
    env_use_gui: Optional[bool] = False
    trial_dir = Path(__file__).parent.parent / "outputs" / "2023-01-18__06-42-48__90D453A7__trial-compound__training-si.100a" / "2023-01-18__06-42-58__94B80ACA__single-trial"
    with load_checkpoint(trial_dir, env_use_gui=env_use_gui) as (algorithm, rllib_env, policy_mapping_dict, net_file_stem):
        episode_end_metrics = run_single_inference(
            algorithm=algorithm, rllib_env=rllib_env, policy_mapping_dict=policy_mapping_dict, custom_callback=None)

    print()
    print(episode_end_metrics)


def test_multiple_inferences():
    trial_dir = Path(__file__).parent.parent / "outputs" / "2023-01-24__19-25-47__67405AA3__trial-compound__training-col8.109b" / "2023-01-24__19-25-57__61C9F929__single-trial"
    df_episode_end_metrics = run_multiple_inferences(trial_dir, n_runs=10)
    print(df_episode_end_metrics)
    for _metric in ("resco_delay", "emissions/CO2"):
        _mu, _std = df_episode_end_metrics[_metric].mean(), df_episode_end_metrics[_metric].std()
        print(f"{_metric}:  µ={_mu:.3f}, std={_std:.4f} ({_std/_mu:.2%})")
    print()

import argparse
import contextlib
import copy
import os
import pickle
import uuid
from datetime import datetime
from pathlib import Path
import shutil
import inspect
from typing import Union, Optional
import tempfile

import ray
import torch.nn
import yaml
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import EnvContext
from ray.tune.registry import register_env
import numpy as np

from environment.actuators import Lane, LaneCompound
from environment.env import GymEnv, RLLibEnv
from environment.wrappers.frame_stack import FrameStack
from environment.wrappers.metrics_wrapper import MetricsWrapper
from environment import env_config_templates
from callbacks.metrics_callbacks import MetricsCallbacks
from util.policy_mapping import AgentSpaces, get_generic_policy_mapping


OUTPUTS_DIR = Path(__file__).parent / "outputs"

N_CPU_CORES = len(os.sched_getaffinity(0))  # Holds the number of CPU cores that we are allowed to use


def _generate_time_uuid_str() -> str:
    _time_str = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    _uuid_str = str(uuid.uuid4())[:8].upper()
    return f"{_time_str}__{_uuid_str}"


def single_train_run(weights_sharing: bool, env_config: GymEnv.Config, n_stacked_frames: int = 1,
                     envs_start_at_random_times: bool = False, trial_suffix: str = None,
                     trials_base_dir: Path = OUTPUTS_DIR,
                     early_stopping: bool = True) -> tuple[str, float]:
    assert n_stacked_frames >= 1
    ray.init(num_gpus=1, num_cpus=N_CPU_CORES-1, local_mode=False)

    # Make sure our "outputs" dir exists
    if not trials_base_dir.exists():
        trials_base_dir.mkdir(parents=False)
    assert trials_base_dir.is_dir()

    # Define the output directory
    _time_uuid_str = _generate_time_uuid_str()
    _trial_name = _time_uuid_str + "__single-trial" + (f"__{trial_suffix}" if trial_suffix is not None else "")
    trial_dir = trials_base_dir / _trial_name
    assert trial_dir.is_dir() is False
    trial_dir.mkdir(parents=True, exist_ok=False)
    ray.rllib.algorithms.algorithm.DEFAULT_RESULTS_DIR = str(trial_dir)  # Hacky way to tell RLLib the log directory
    checkpoints_dir = trial_dir / "checkpoints"

    # Env construction
    env_base_seed = int(np.random.randint(low=0, high=2 ** 16 - 1))

    def _create_env(env_context: Optional[EnvContext]):  # --> env_context will only be None when called from below!
        _env_config = copy.deepcopy(env_config)
        if env_context is not None:
            _env_index = env_context.num_workers * env_context.vector_index + env_context.worker_index  # Index starts 1
            _is_wrapped_for_logging = (_env_index == 1) or not envs_start_at_random_times  # If no random times,wrap all
            _env_config.seed = env_base_seed + _env_index
            # if _env_index == 1:
            #     _env_config.use_gui = True
            if envs_start_at_random_times and _env_index != 1:
                # All non-CSV/YAML-logging envs may start from different times to simulate a broader range of traffic
                # types (see *.rou.xml file). It might boost learning performance, as the observations are more diverse.
                _max_time = \
                    _env_config.simulation_end_time - (_env_config.simulation_end_time-_env_config.simulation_begin_time)/4
                _env_config.simulation_begin_time = np.random.randint(_env_config.simulation_begin_time, _max_time)
            gym_env = GymEnv(config=_env_config, do_output_info=False)
            if _is_wrapped_for_logging:
                _env_index_log_dir = trial_dir / f"logs_env-index-{_env_index}"
                _env_index_log_dir.mkdir(parents=False, exist_ok=False)
                gym_env = MetricsWrapper(
                    env=gym_env, episode_end_yaml_file_path=_env_index_log_dir / "episode-end-metrics.yaml",
                    episode_end_trip_info_objects_pkl_path=None,  # _env_index_log_dir / "episode-end-trip-info-objects.pkl",
                    intra_episode_csv_file_path=None,  # _env_index_log_dir / "intra-episode-metrics.csv",
                    intra_episode_logging_frequency=100)
        else:
            gym_env = GymEnv(config=_env_config, do_output_info=False)
        if n_stacked_frames >= 2:
            gym_env = FrameStack(env=gym_env, n_frames=n_stacked_frames)
        return RLLibEnv(env=gym_env)
    register_env(name="_just_some_name_", env_creator=_create_env)

    # Save train config, train script & calling script
    env_config.save_to_yaml(yaml_path=trial_dir / "env-config.yaml")
    shutil.copy(__file__, trial_dir / Path(__file__).name)
    _frame = inspect.stack()[1]
    _module = inspect.getmodule(_frame[0])
    if _module.__file__ != __file__ and Path(_module.__file__).stem != "process":  # In multiprocessing, skip saving
        shutil.copy(_module.__file__, trial_dir / Path(_module.__file__).name)

    # Determine action and observation spaces by taking a look into the environment
    with contextlib.closing(_create_env(env_context=None)) as _template_env:
        _action_spaces = _template_env.action_space
        _observation_spaces = _template_env.observation_space
        _agent_spaces = {n: AgentSpaces(observation_space=_observation_spaces[n], action_space=_action_spaces[n])
                         for n in _action_spaces}
        policy_specs, _, policy_mapping_dict = get_generic_policy_mapping(agent_spaces=_agent_spaces,
                                                                          weights_sharing=weights_sharing)

    def _policy_mapping_fn(_agent_id, *_args, **_kwargs):
        assert _agent_id in policy_mapping_dict
        return policy_mapping_dict[_agent_id]

    # Train
    algorithm_config = {
        "multiagent": {
            # "policies": {
            #     '0': (PPOTorchPolicy, _single_observation_space, _single_action_space, {})
            # },
            # "policy_mapping_fn": (lambda id: '0'),  # Traffic lights are always controlled by this policy
            "policies": policy_specs,
            "policy_mapping_fn": _policy_mapping_fn,
            "count_steps_by": "env_steps",  # default: "env_steps",
            "policy_map_cache": tempfile.mkdtemp(),
        },
        # "no_done_at_end": True,
        # "soft_horizon": True,
        "framework": "torch",
        "model": {
            "fcnet_hiddens": [256, 256, 256, 256],  # default: [256, 256]
            "fcnet_activation": torch.nn.LeakyReLU,
            # "vf_share_layers": True,  # default: False

            # --- Attention Nets ---
            # "use_attention": True,  # default: False
            # "attention_num_transformer_units": 3,  # default: 1
            # "attention_dim": 64,  # default: 64
            # "attention_num_heads": 2,  # default: 1
            # "attention_head_dim": 32,  # default: 32
            # "attention_memory_inference": 50,  # default: 50
            # "attention_memory_training": 50,  # default: 50
            # "attention_position_wise_mlp_dim": 32,  # default: 32
            # "attention_init_gru_gate_bias": 2.0,  # default: 2.0
            # "attention_use_n_prev_actions": 0,  # default: 0
            # "attention_use_n_prev_rewards": 0,  # default: 0
        },
        # "output": str(trial_dir),  # Enable this to store experiences into JSON files
        "callbacks": MetricsCallbacks,
        "num_workers": 5,
        # "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
        # "num_gpus": 1,
        # "num_gpus_per_worker": 0.1,

        # "evaluation_interval": 2,
        # "evaluation_duration": 1,

        # Cope with failures within the workers
        "ignore_worker_failures": True,
        # "recreate_failed_workers": True,  # --> Not really helpful in our case, as MetricsWrapper prevents recreation

        # Explanations for the params below:    https://docs.ray.io/en/master/rllib/rllib-sample-collection.html
        "train_batch_size": 4000,  # default: 4000
        "rollout_fragment_length": 200,  # default: 200
        # "batch_mode": "complete_episodes",  # default: "truncated_episodes"
        # "num_sgd_iter": 5,  # default: 30
        # "sgd_minibatch_size": 500,  # default: 128

        "lr": 1e-5,  # default: 5e-5
        # "lr_schedule": [
        #     [0, 1e-5],
        #     [500_000, 1e-6]
        # ],
        # "gamma": 0.99,  # default: 0.99
        # "lambda": 0.99,  # default: 1.0
        # "entropy_coeff": 0.09,  # default: 0
        # "vf_loss_coeff": 0.04,  # default: 1.0
        # "clip_param": 0.3,  # default: 0.3
        # "grad_clip": 10,  # default: 10.0
        # "use_kl_loss": False,
        # "kl_coeff": 0.2,  # default: 0.2
    }
    algorithm = PPO(env="_just_some_name_", config=algorithm_config)
    with open(trial_dir / "algorithm-construction-params.pkl", mode="wb") as _file:
        _algorithm_config_copy = copy.deepcopy(algorithm_config)
        _algorithm_config_copy["multiagent"]["policy_mapping_fn"] = "//to be set//"
        pickle.dump({"algorithm_type": type(algorithm),
                     "algorithm_config": _algorithm_config_copy,
                     "policy_mapping_dict": policy_mapping_dict,
                     "n_stacked_frames": n_stacked_frames}, _file)

    n_max_episodes = 1400

    # Train loop
    _best_resco_delay = float("+inf")
    _first_resco_delay = None
    _n_iterations_since_last_improvement = 0
    _i_iteration = 0
    _checkpoint_dir = None
    while True:
        _i_iteration += 1
        started_at = datetime.now()
        results_dict = algorithm.train()
        # print(results_dict)
        print(f"[Ep. {results_dict['episodes_total']}/{n_max_episodes}] -- Iteration took {(datetime.now() - started_at).total_seconds():.1f}s")
        if "resco_delay_mean" in results_dict["custom_metrics"]:
            _n_iterations_since_last_improvement += 1
            _resco_delay_mean = results_dict["custom_metrics"]["resco_delay_mean"]
            if _first_resco_delay is None:
                _first_resco_delay = _resco_delay_mean
            if _resco_delay_mean < _best_resco_delay:
                _best_resco_delay = _resco_delay_mean
                _n_iterations_since_last_improvement = 0
                if _checkpoint_dir is not None:
                    shutil.rmtree(_checkpoint_dir)  # Delete previous checkpoint to save memory
                    _checkpoint_dir.mkdir(parents=False)  # Recreate empty dir, so we don't confuse below cp creation
                _checkpoint_dir = Path(algorithm.save(checkpoint_dir=str(checkpoints_dir), prevent_upload=True))
                print(f" --> New hi-score: {_best_resco_delay:.1f}; saving model checkpoint '{_checkpoint_dir.name}'")
            # Early stopping
            if _resco_delay_mean >= _first_resco_delay and _n_iterations_since_last_improvement >= 30:
                break
            if early_stopping and _resco_delay_mean < _first_resco_delay and _n_iterations_since_last_improvement >= 70:
                break
        # Clip episodes number
        if results_dict["episodes_total"] > n_max_episodes:
            break
    # trainer.evaluate()
    algorithm.stop()  # Closes all envs and saves possibly open files (e.g. those in MetricsWrapper)
    return trial_dir.name, float(_best_resco_delay)


def multiple_parallel_train_runs(n_parallel_runs: int = 3, trial_compound_dir_suffix: str = None, **kwargs) -> float:
    """Calls the above function "single_train_run()" multiple times in parallel, each with the exact same config"""
    from multiprocessing import set_start_method
    from concurrent.futures import ProcessPoolExecutor

    # Generate a compound directory that encloses all single trials
    _time_uuid_str = _generate_time_uuid_str()
    _trial_compound_dir_name = _time_uuid_str + "__trial-compound" + \
                  (f"__{trial_compound_dir_suffix}" if trial_compound_dir_suffix is not None else "")
    _trial_compound_dir = OUTPUTS_DIR / _trial_compound_dir_name
    _trial_compound_dir.mkdir(parents=False, exist_ok=False)
    kwargs["trials_base_dir"] = _trial_compound_dir

    print(f"Starting compound trainings at:  '{_trial_compound_dir.expanduser().absolute()}'")

    set_start_method("spawn")
    with ProcessPoolExecutor(max_workers=n_parallel_runs) as executor:
        _futures = [executor.submit(single_train_run, **kwargs) for _ in range(n_parallel_runs)]
        _results = [f.result() for f in _futures]
        subfolder_names__result_scores: dict[str, float] = {subfolder_name: score for subfolder_name, score in _results}
    with open(_trial_compound_dir / "result_scores.yaml", mode="w", encoding="utf-8") as file:
        yaml.dump(data=subfolder_names__result_scores, stream=file)
    return min(subfolder_names__result_scores.values())


if __name__ == '__main__':
    names__functions = env_config_templates.FUNCTION_NAMES__TEMPLATE_FUNCTIONS

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--env_name", default="sumo_rl_single_intersection__low_traffic", type=str, choices=names__functions.keys())
    _parser.add_argument("--experiment_name", type=str, default="training_si-lt.1")
    _parser.add_argument("--use_v2i", action="store_true")
    _parser.add_argument("--use_speed_control", action="store_true")
    args = _parser.parse_args()

    print(f"Num CPU cores available: {N_CPU_CORES}")
    print()
    print("Commandline args:")
    print("-----------------")
    print(yaml.dump(args.__dict__))

    # Instantiate the environment config and set train params
    env_config: GymEnv.Config = names__functions[args.env_name]()
    if args.use_speed_control is True:
        env_config.lane_compound_config = LaneCompound.Config(reward_type="time-spent-on-lanes")

    env_config.trafficlight_config.observations_add_leading_vehicle = args.use_v2i  # --> Enables V2I (Vehicle-to-Infrastructure communication)
    env_config.trafficlight_config.reward_type = "time-spent-on-lanes"

    if args.use_speed_control:
        # env_config.lane_config = Lane.Config(reward_type="time-spent-on-lanes")
        env_config.lane_compound_config = LaneCompound.Config(reward_type="time-spent-on-lanes")  # --> Using LaneCompound instead of Lane renders helpful in case of numerous controlled lanes

    kwargs = {
        "weights_sharing": False,
        "env_config": env_config,
        "trial_suffix": None,
        "early_stopping": True,
    }
    # single_train_run(**kwargs)
    best_score = multiple_parallel_train_runs(n_parallel_runs=5,
                                              trial_compound_dir_suffix=args.experiment_name,
                                              **kwargs)
    print()
    print(f"Best score: {best_score:.2f}")

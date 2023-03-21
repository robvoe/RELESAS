from typing import Dict, Union, Optional

from ray.rllib import RolloutWorker, Policy, BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

from environment.env import GymEnv
from util.nested_dicts import flatten_nested_dict


class MetricsCallbacks(DefaultCallbacks):
    # def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy],
    #                      episode: Union[Episode, EpisodeV2], env_index: Optional[int] = None, **kwargs) -> None:
    #     # Make sure this episode has just been started (only initial obs
    #     # logged so far).
    #     assert episode.length == 0, (
    #         "ERROR: `on_episode_start()` callback should be called right "
    #         "after env reset!"
    #     )
    #     print(f"Episode {episode.episode_id} starts.   worker_index = {worker.worker_index}, env_index = {env_index}")
    #
    # def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
    #                     policies: Optional[Dict[PolicyID, Policy]] = None, episode: Union[Episode, EpisodeV2],
    #                     env_index: Optional[int] = None, **kwargs) -> None:
    #     # Make sure this episode is ongoing.
    #     assert episode.length > 0, (
    #         "ERROR: `on_episode_step()` callback should not be called right "
    #         "after env reset!"
    #     )
    #     episode.user_data["pole_angles"] += [2]
    #     if episode.length % 100 == 0:
    #         _env: GymEnv = base_env.get_sub_environments()[env_index].unwrapped
    #         episode.user_data["n_vehicles_enqueued"] += [_env.get_intra_episode_metrics()["n_vehicles_enqueued"]]
    #
    #     # pole_angle = abs(episode.last_observation_for()[2])
    #     # raw_angle = abs(episode.last_raw_obs_for()[2])
    #     # assert pole_angle == raw_angle
    #     # episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                       episode: Union[Episode, EpisodeV2, Exception], env_index: Optional[int] = None,
                       **kwargs) -> None:
        # Check if there are multiple episodes in a batch, i.e. "batch_mode": "truncate_episodes".
        # if worker.policy_config["batch_mode"] == "truncate_episodes":
        #     # Make sure this episode is really done.
        #     assert episode.batch_builder.policy_collectors["default_policy"].batches[-1]["dones"][-1], (
        #         "ERROR: `on_episode_end()` should only be called "
        #         "after episode is done!"
        #     )
        # print(f"Episode {episode.episode_id} ended.   worker_index = {worker.worker_index}, env_index = {env_index}")
        # _is_wrapped_for_logging = worker.worker_index == 1 and env_index == 0  # --> Same as in train*.py
        # if _is_wrapped_for_logging:
        _env: GymEnv = base_env.get_sub_environments()[env_index].unwrapped
        _episode_end_metrics = _env.get_episode_end_metrics()
        if _episode_end_metrics["done"] is True:
            del _episode_end_metrics["done"]
            # Add metrics to 'custom_metrics' structure. Flatten nested dicts, if necessary (e.g. emissions-subdict)
            _flattened_dict = flatten_nested_dict(_episode_end_metrics)
            episode.custom_metrics.update(_flattened_dict)

    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))
    #
    # def on_train_result(self, *, algorithm: Optional[Algorithm] = None, result: dict, trainer=None, **kwargs, ) -> None:
    #     print(
    #         "Algorithm.train() result: {} -> {} episodes".format(
    #             algorithm, result["episodes_this_iter"]
    #         )
    #     )
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True
    #
    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
    #     result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     print(
    #         "policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #             policy, result["sum_actions_in_train_batch"]
    #         )
    #     )
    #
    # def on_postprocess_trajectory(self, *, worker: RolloutWorker, episode: Episode, agent_id: str, policy_id: str,
    #                               policies: Dict[str, Policy], postprocessed_batch: SampleBatch,
    #                               original_batches: Dict[str, Tuple[Policy, SampleBatch]], **kwargs):
    #     pass
    #     # print("postprocessed {} steps".format(postprocessed_batch.count))
    #     # if "num_batches" not in episode.custom_metrics:
    #     #     episode.custom_metrics["num_batches"] = 0
    #     # episode.custom_metrics["num_batches"] += 1

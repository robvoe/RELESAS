"""
This script loads EpisodeEndMetrics YAML files that are constantly written during training. It can be used to
visualize model performance over training time. The script was organized in such a way, that plotting functions
were implemented as PyTest test functions. Nevertheless, the functions can alternatively be called from normal scripts.
"""
from collections import defaultdict
from pathlib import Path
from typing import Optional, Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle
from tabulate import tabulate

from environment.env import GymEnv
from util.finished_trainings.episode_end_yamls import load_single_trial_results
from util.finished_trainings.path_lookup import get_single_trial_dirs


sns.set(style='darkgrid', rc={'figure.figsize': (7.2, 4.45),
                            'xtick.labelsize': 16,
                            'ytick.labelsize': 16,
                            'font.size': 15,
                            'figure.autolayout': True,
                            'axes.titlesize' : 16,
                            'axes.labelsize' : 17,
                            'lines.linewidth' : 2,
                            'lines.markersize' : 6,
                            # "ytick.left": True,
                            'legend.fontsize': 15})
colors = sns.color_palette("colorblind", 10)
#colors = sns.color_palette("Set1", 2)
#colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
dashes_styles = cycle(['-', '-.', '--', ':'])
sns.set_palette(colors)
colors = cycle(colors)


def _fuse_multiple_trials(single_trial_paths: list[Path],
                          moving_average_window_size: int = None,
                          considered_metrics: list[str] = None) -> pd.DataFrame:
    """
    Loads multiple trials and fuses them into a single DataFrame. The index of the returned DataFrame is purely
    synthetic - so it carries no timestamp information.
    """
    # Load the given single trials
    _trial_dfs = [load_single_trial_results(p) for p in single_trial_paths]

    if considered_metrics is not None:
        assert all(all(m in df.columns for df in _trial_dfs) for m in considered_metrics), \
            f"Not all given metrics {considered_metrics} exist in all of the trials!"
        _trial_dfs = [p[considered_metrics] for p in _trial_dfs]

    # Find the shortest df-length and truncate all others to its size
    _minimum_len = min(len(df) for df in _trial_dfs)
    _trial_dfs = [df.iloc[:_minimum_len] for df in _trial_dfs]
    _fused_df = pd.DataFrame(index=pd.RangeIndex(start=0, stop=_minimum_len))

    # Now generate mean/median/std over the values
    _metric_names = [c for c in _trial_dfs[0].columns]
    for _metric in _metric_names:
        _stacked_values = np.stack([df[_metric].values for df in _trial_dfs], axis=0)
        _fused_df[_metric + "_mean"] = _stacked_values.mean(axis=0)
        _fused_df[_metric + "_median"] = np.median(_stacked_values, axis=0)
        _fused_df[_metric + "_std"] = _stacked_values.std(axis=0)
        _fused_df[_metric + "_min"] = np.min(_stacked_values, axis=0)

    if moving_average_window_size is not None:
        _fused_df = _fused_df.rolling(window=moving_average_window_size).mean()
    return _fused_df


def plot_multiple_trials_in_temporal_order(
        folders: list[Path],
        folders_type: Literal["single trials", "trial compounds"] = "single trials",
        labels: list[str] = None, plot_metrics: Iterable[str] =
            ("resco_delay", "n_finished_trips", "emissions/CO", "emissions/CO2", "emissions/HC",
             "emissions/PMx", "emissions/NOx", "emissions/fuel"),
        metric_labels: Sequence[str] =
            ("Trip delay [s]", "Number finished trips", "Mean trip CO emission [mg]", "Mean trip CO₂ emission [mg]",
             "Mean trip HC emission [mg]", "Mean trip PMₓ emission [mg]", "Mean trip NOₓ emission [mg]",
             "Mean trip fuel consumption [ml]"),
        moving_average_window_size: int = 5,
        mark_min_max: bool = True,
        drop_first_n_episodes: int = None,
        drop_after_nth_episode: int = None,
        title: Optional[str] = "") -> None:
    plot_metrics = list(plot_metrics)
    assert len(folders) > 0
    assert all(f.is_dir() for f in folders), f"At least one of the {len(folders)} given folders does not exist!"
    assert folders_type in ("single trials", "trial compounds")
    if labels is None:
        labels = [None] * len(folders)
    assert len(folders) == len(labels)
    assert len(plot_metrics) == len(metric_labels)
    assert len(metric_labels) == len(set(metric_labels)), f"No duplicates allowed in labels!"

    _metric_min_max_function = {"resco_delay": np.min, "n_finished_trips": np.max, "emissions/CO": np.min,
                                "emissions/CO2": np.min, "emissions/HC": np.min, "emissions/PMx": np.min,
                                "emissions/NOx": np.min, "emissions/fuel": np.min}

    # Load data from all given folders
    folders__dataframes: dict[Path, pd.DataFrame] = {}
    if folders_type == "single trials":
        for _trial_path in folders:
            _df_trial = load_single_trial_results(_trial_path, moving_average_window_size=moving_average_window_size)
            folders__dataframes[_trial_path] = _df_trial
    elif folders_type == "trial compounds":
        for _compound_path in folders:
            _single_trial_paths = get_single_trial_dirs(trial_compound_dir=_compound_path)
            assert len(_single_trial_paths) > 0, f"Given compound path {_compound_path} contains no trials."
            _df_fused_trials = _fuse_multiple_trials(
                single_trial_paths=_single_trial_paths, considered_metrics=plot_metrics,
                moving_average_window_size=moving_average_window_size)
            folders__dataframes[_compound_path] = _df_fused_trials
    else:
        raise RuntimeError("We should not have ended-up here!")

    # # Hacky way to add another timeline to the plot
    # means = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # stds = [0.1] * 9
    # assert len(means) == len(stds)
    # _df: pd.DataFrame = sorted(folders__dataframes.values(), key=lambda d: len(d), reverse=True)[0].copy()
    # if len(_df) < len(means):
    #     _last_index = _df.index[0] + (_df.index[-1]-_df.index[0]) / len(_df) * len(means)
    #     _step = (_last_index - _df.index[0]) / len(means)
    #     _index = pd.Index(data=np.arange(_df.index[0], _last_index, step=_step), dtype=pd.Float64Dtype)
    # else:
    #     _index = _df.index[:len(means)]
    # _df = pd.DataFrame(index=_index)
    # for _metric in plot_metrics:
    #     _df[_metric + "_mean"] = pd.Series(means, index=_index).rolling(window=moving_average_window_size).min()
    #     _df[_metric + "_std"] = pd.Series(stds, index=_index).rolling(window=moving_average_window_size).min()
    # folders__dataframes["no-real-path"] = _df
    # if labels is not None:
    #     labels += ["RESCO"]

    # Plot :-)
    _env_config_file_path = folders[0] / "env-config.yaml"
    if title is None and _env_config_file_path.is_file():
        env_config = GymEnv.Config.load_from_yaml(_env_config_file_path)
        title = env_config.net_file_stem

    metric__best_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(defaultdict)  # maps  metric --> labels --> min/mean/std

    for _metric, _metric_label in zip(plot_metrics, metric_labels):
        # if _metric == "resco_delay":
        #     plt.yscale("log")
        #     plt.tick_params(axis="y", which="minor")
        #     plt.rcParams["ytick.left"] = True
        # else:
        #     plt.rcParams["ytick.left"] = False
        plt.figure(figsize=(9, 7))
        for _df, _trial_label in zip(folders__dataframes.values(), labels):
            _mean_vals = _df[_metric + "_mean"].values
            _std_vals = _df[_metric + "_std"].values
            _min_vals = _df[_metric + "_min"].values
            if drop_first_n_episodes is not None:
                _mean_vals = _mean_vals[drop_first_n_episodes:]
                _std_vals = _std_vals[drop_first_n_episodes:]
            if drop_after_nth_episode is not None:
                _mean_vals = _mean_vals[:drop_after_nth_episode]
                _std_vals = _std_vals[:drop_after_nth_episode]
            _high, _low = _mean_vals + _std_vals, _mean_vals - _std_vals
            _drawn_line, = plt.plot(_mean_vals, label=_trial_label)
            plt.fill_between(range(len(_mean_vals)), _low, _high, alpha=0.4)
            # Determine best scores and gather them in one place
            _cleaned_mean_vals, _cleaned_std_vals, _cleaned_min_vals = \
                _mean_vals[~np.isnan(_mean_vals)], _std_vals[~np.isnan(_std_vals)], _min_vals[~np.isnan(_min_vals)]
            _best_mean, _best_min = \
                _metric_min_max_function[_metric](_cleaned_mean_vals), _metric_min_max_function[_metric](_cleaned_min_vals)
            _idx_of_best_mean = np.where(_mean_vals == _best_mean)[0][-1]
            _std_of_best_mean = _std_vals[_idx_of_best_mean]
            metric__best_scores[_metric][_trial_label] = {"mean": _best_mean, "std": _std_of_best_mean, "min": _best_min}
            # If necessary, annotate the plot
            if mark_min_max is True:
                _rgba = (*_drawn_line.get_color(), 0.2)   # original value: "0.9"
                plt.annotate(f"{_best_mean:.2f} (±{_std_of_best_mean:.2f})", xy=(_idx_of_best_mean, _best_mean), xycoords='data',
                             bbox=dict(boxstyle="round4,pad=.5", fc=_rgba), xytext=(40, -22 if _trial_label == "Model 1" else +40),
                             textcoords='offset points', ha='center',
                             arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=0.2", color="b"))
                # plt.axhline(y=_min_max, linestyle="--", alpha=0.7)
                # plt.axvline(x=_idx, linestyle="--", alpha=0.7)
                # _ytick_locs, _ = plt.yticks()
                # _ytick_locs[-1] = _min_max
                # plt.yticks(_ytick_locs)
        plt.xlabel("Training episode")
        plt.ylabel(_metric_label)
        plt.suptitle(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Now print the training scores (best scores per trial or trial compound)
    print("\n\n")
    for _metric, _metric_label in zip(plot_metrics, metric_labels):
        _df_scores = pd.DataFrame(metric__best_scores[_metric]).transpose()
        for _c in _df_scores.columns:
            _df_scores[f"{_c}/1000"] = _df_scores[_c] / 1000
        print(f"\n\n{_metric_label}:")
        print(tabulate(_df_scores, headers="keys", floatfmt=".2f"))


def test_plot_exemplary_trainings():
    # --> SingleIntersection
    # folder1 = Path(__file__).parent / "outputs" / "2023-01-18__16-17-50__B0F574AB__trial-compound__single-intersection__traffic-light-only"
    # folder2 = Path(__file__).parent / "outputs" / "2023-01-24__18-44-57__B28C526D__trial-compound__single-intersection__traffic-light_v2i"
    # folder3 = Path(__file__).parent / "outputs" / "2023-01-21__11-28-25__B7135A0E__trial-compound__single-intersection__traffic-light_v2i_speed-advice"
    # --> SingleIntersection (low traffic)
    # folder1 = Path(__file__).parent / "outputs" / "2023-01-20__10-27-50__8A8B9312__trial-compound__single-intersection-low-traffic__traffic-light-only"
    # folder2 = Path(__file__).parent / "outputs" / "2023-01-24__18-50-02__7D359D1F__trial-compound__single-intersection-low-traffic__traffic-light_v2i"
    # folder3 = Path(__file__).parent / "outputs" / "2023-01-21__12-18-42__20B63E94__trial-compound__single-intersection-low-traffic__traffic-light_v2i_speed-advice"
    # --> SingleIntersection (very low traffic)
    folder1 = Path(__file__).parent / "outputs" / "2023-01-20__13-41-05__0DCF9792__trial-compound__single-intersection-very-low-traffic__traffic-light-only"
    folder2 = Path(__file__).parent / "outputs" / "2023-01-24__18-55-09__6FCD6A79__trial-compound__single-intersection-very-low-traffic__traffic-light_v2i"
    folder3 = Path(__file__).parent / "outputs" / "2023-01-21__13-19-11__1F199778__trial-compound__single-intersection-very-low-traffic__traffic-light_v2i_speed-advice"
    plot_multiple_trials_in_temporal_order(
        folders=[folder1, folder2, folder3],
        folders_type="trial compounds",
        labels=["TL only", "TL + V2I", "TL + V2I + SpeedAdvice"],
        plot_metrics=["resco_delay", "emissions/CO2"],
        metric_labels=["Trip delay [s]", "Mean trip CO₂ emission [mg]"],
        moving_average_window_size=15,
        mark_min_max=True,
        drop_first_n_episodes=40,
        drop_after_nth_episode=None,
        title="")

"""
This file loads a trained model and runs one episode of its environment. Meanwhile, data is collected and
eventually plotted.
The script was organized in such a way, that data collection runs separately from plotting - instead, collected data
is cached to disk, and later loaded by plotting methods. Data collection & plotting methods were implemented as PyTest
test functions.
"""
import pickle
from collections import defaultdict
from itertools import cycle, islice
from pathlib import Path
from typing import Optional, cast, Union, Any, Tuple, List, Literal

import matplotlib
import matplotlib.patches
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import seaborn as sns
import scipy.interpolate
from ray.rllib.utils.typing import MultiAgentDict

from environment.env import GymEnv
from environment.actuators import TrafficLight, Lane, LaneCompound, BaseActuator, lane_util
from environment.util.trip_info import TripInfo
from util.finished_trainings.evaluation import load_checkpoint, run_single_inference

colors = sns.color_palette("colorblind", 10)
#colors = sns.color_palette("Set1", 2)
#colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
dashes_styles = cycle(['-', '-.', '--', ':'])
sns.set_palette(colors)
colors = cycle(colors)


CACHE_FILE_PATH = Path(__file__).parent / f"{Path(__file__).stem}_cache.pkl"


def _set_fig_style():
    sns.set(style='darkgrid', rc={'figure.figsize': (7.2, 4.45),
                                  'xtick.labelsize': 16,
                                  'ytick.labelsize': 16,
                                  'font.size': 15,
                                  'figure.autolayout': True,
                                  'axes.titlesize': 16,
                                  'axes.labelsize': 17,
                                  'lines.linewidth': 2,
                                  'lines.markersize': 6,
                                  # "ytick.left": True,
                                  'legend.fontsize': 15})


def _assert_convert_action(action) -> list[Union[float, int]]:
    assert isinstance(action, (float, int, np.ndarray, np.int, np.int32, np.int64)), \
        f"Invalid action type: {type(action).__class__.__name__}"
    if isinstance(action, np.ndarray):
        assert action.ndim == 1
        if action.dtype in (np.int, np.int16, np.int32, np.int64):
            action = [int(a) for a in action]
        elif action.dtype in (np.float, np.float16, np.float32, np.float64):
            action = [float(a) for a in action]
        else:
            raise ValueError(f"Invalid action type: {type(action.dtype).__name__}")
    elif isinstance(action, (np.int, np.int32, np.int64)):
        action = [int(action)]
    assert isinstance(action, list), "Seems I forgot to consider some type conversion.."
    return action


def test_generate_inference_statistics_cache():
    """
    This function runs a loaded model in inference mode and collects statistics. After finishing, the collected
    data will be cached to "plot_trained_model_inference_cache.pkl" - which in turn can be utilized by the other
    functions within this module.
    """
    single_trial_dir = Path(__file__).parent / "outputs" / "2023-01-23__03-37-20__AA27092D__trial-compound__training-gr.108j" / "2023-01-23__03-37-30__A746C964__single-trial*"

    env_use_gui: Optional[bool] = None
    with load_checkpoint(single_trial_dir, env_use_gui=env_use_gui, env_allow_skip_idle_time_steps=False) as (_algorithm, _rllib_env, _policy_mapping_dict, net_file_stem):
        _gym_env = cast(GymEnv, _rllib_env.unwrapped)
        _lane_compound_actuators: List[LaneCompound] = [a for a in _gym_env.actuators.values() if isinstance(a, LaneCompound)]
        _lane_actuators: List[Lane] = [a for a in _gym_env.actuators.values() if isinstance(a, Lane)]

        scenario_lane_ids: list[str] = [l for l in _gym_env._sumo_connection.lane.getIDList() if not l.startswith(":")]
        _traffic_light_ids__actions_over_time: dict[str, dict[int, Union[float, int]]] = defaultdict(defaultdict)
        _lane_ids__actions_over_time: dict[str, dict[int, Union[float, int]]] = defaultdict(defaultdict)
        _lane_ids__normalized_speed_limits_over_time: dict[str, dict[int, float]] = defaultdict(defaultdict)
        _vehicle_ids__driven_meters_over_time: dict[str, dict[int, float]] = defaultdict(defaultdict)
        _vehicle_ids__normalized_speed_limit_over_time: dict[str, dict[int, float]] = defaultdict(defaultdict)  # normalized by the current lane limit
        _vehicle_ids__current_speed_over_time: dict[str, dict[int, float]] = defaultdict(defaultdict)
        _lane_ids__mean_speed_over_time: dict[str, dict[int, float]] = defaultdict(defaultdict)
        lane_ids__vehicle_ids: dict[str, set[str]] = defaultdict(set)
        vehicle_ids__route_ids: dict[str, str] = {}
        lane_ids__mean_vehicle_waiting_times: dict[str, float] = defaultdict(float)
        lane_ids__vehicle_wise_usage_times: dict[str, dict[str, float]] = defaultdict(dict)  # Bases on "time spent on lanes"

        is_speed_advice: bool = _gym_env.config.lane_config is not None or _gym_env.config.lane_compound_config is not None
        subject_to_speed_limit = None if not is_speed_advice else \
            (_gym_env.config.lane_config.subject_to_speed_limit if _gym_env.config.lane_config is not None else _gym_env.config.lane_compound_config.subject_to_speed_limit)

        def _data_collection_callback(_observations: MultiAgentDict, _actions: MultiAgentDict) -> None:
            # --> Log actions
            _sim_time = int(_gym_env._sumo_connection.simulation.getTime())
            for _agent_id, _agent_action in _actions.items():
                _actuator: BaseActuator = _gym_env.actuators[_agent_id]
                _converted_agent_action_list = _assert_convert_action(_agent_action)
                if isinstance(_actuator, TrafficLight):
                    assert len(_converted_agent_action_list) == 1
                    _traffic_light_ids__actions_over_time[_actuator.id][_sim_time] = _converted_agent_action_list[0]
                elif isinstance(_actuator, Lane):
                    assert len(_converted_agent_action_list) == 1
                    _lane_ids__actions_over_time[_actuator.id][_sim_time] = _converted_agent_action_list[0]
                elif isinstance(_actuator, LaneCompound):
                    assert len(_converted_agent_action_list) == len(_actuator.controlled_lane_ids)
                    for _lane_id, _lane_action in zip(_actuator.controlled_lane_ids, _converted_agent_action_list):
                        _lane_ids__actions_over_time[_lane_id][_sim_time] = _lane_action
                else:
                    raise RuntimeError("We should not have ended-up here!")

            # --> Log simulation state (lane speed limit, driven vehicle distance over time, vehicle speed, etc.)
            if is_speed_advice is True:
                _current_lane_speed_limits_norm: dict[str, float] = _gym_env.get_current_lane_speed_limits_norm()
                for _lane_id, _speed_limit in _current_lane_speed_limits_norm.items():
                    _lane_ids__normalized_speed_limits_over_time[_lane_id][_sim_time] = _speed_limit

            for _lane_id in scenario_lane_ids:
                _mean_speed = _gym_env._sumo_connection.lane.getLastStepMeanSpeed(_lane_id)  # Equals max speed if no vehicles present
                _n_halting_vehicles = _gym_env._sumo_connection.lane.getLastStepHaltingNumber(_lane_id)
                _lane_ids__mean_speed_over_time[_lane_id][_sim_time] = _mean_speed
                lane_ids__mean_vehicle_waiting_times[_lane_id] = _n_halting_vehicles
                lane_ids__vehicle_wise_usage_times[_lane_id].update(_gym_env._lanes__vehicle_usage_times[_lane_id])

            _vehicle_ids = _gym_env._sumo_connection.vehicle.getIDList()
            for _vehicle_id in _vehicle_ids:
                _lane_id: str = _gym_env._sumo_connection.vehicle.getLaneID(_vehicle_id)
                if _lane_id in scenario_lane_ids:
                    lane_ids__vehicle_ids[_lane_id].add(_vehicle_id)
                _vehicle_ids__driven_meters_over_time[_vehicle_id][_sim_time] = \
                    _gym_env._sumo_connection.vehicle.getDistance(_vehicle_id)
                _vehicle_ids__current_speed_over_time[_vehicle_id][_sim_time] = \
                    _gym_env._sumo_connection.vehicle.getSpeed(_vehicle_id)
                if _vehicle_id not in vehicle_ids__route_ids:
                    vehicle_ids__route_ids[_vehicle_id] = _gym_env._sumo_connection.vehicle.getRouteID(_vehicle_id)
                if is_speed_advice is True:
                    _normalized_lane_speed_limit = 1 if _lane_id not in _current_lane_speed_limits_norm else _current_lane_speed_limits_norm[_lane_id]
                    _speed_limit_applies_to_vehicle = False
                    if subject_to_speed_limit == "whole-lane":
                        _speed_limit_applies_to_vehicle = True
                    elif subject_to_speed_limit == "leading-vehicle":
                        if _gym_env.config.lane_config is not None:
                            _speed_limit_applies_to_vehicle = any(a._current_leading_vehicle_id == _vehicle_id for a in _lane_actuators)
                        elif _gym_env.config.lane_compound_config is not None:
                            _speed_limit_applies_to_vehicle = any(_vehicle_id in a._lane_ids__current_leading_vehicle_id.values() for a in _lane_compound_actuators)
                    _vehicle_ids__normalized_speed_limit_over_time[_vehicle_id][_sim_time] = _normalized_lane_speed_limit if _speed_limit_applies_to_vehicle else 1

        episode_end_metrics = run_single_inference(algorithm=_algorithm, policy_mapping_dict=_policy_mapping_dict,
                                                   rllib_env=_rllib_env, custom_callback=_data_collection_callback)

        for _lane_id in scenario_lane_ids:
            lane_ids__mean_vehicle_waiting_times[_lane_id] /= _gym_env.config.simulation_duration
        lane_ids__mean_usage_time_per_vehicle = \
            {l: (sum(v.values())/len(v) if len(v) else 0) for l, v in lane_ids__vehicle_wise_usage_times.items()}

        # Convert the collected data to DataFrames
        lane_ids__max_speed: dict[str, float] = {l: _gym_env._sumo_connection.lane.getMaxSpeed(l) for l in scenario_lane_ids}
        edge_ids__lane_ids: dict[str, list[str]] = lane_util.cluster_lanes_by_edge(sumo_connection=_gym_env._sumo_connection, lane_ids=scenario_lane_ids)
        vehicle_ids__route_ids: dict[str, str] = vehicle_ids__route_ids  # Just to show that we have this :-)
        trip_info_objects: list[TripInfo] = _gym_env.trip_info_objects
        df__traffic_light_ids__actions_over_time = pd.DataFrame.from_dict(_traffic_light_ids__actions_over_time).sort_index()
        df__lane_ids__actions_over_time = pd.DataFrame.from_dict(_lane_ids__actions_over_time).sort_index()
        df__lane_ids__normalized_speed_limits_over_time = pd.DataFrame.from_dict(_lane_ids__normalized_speed_limits_over_time).sort_index()
        df__lane_ids__mean_speed_over_time = pd.DataFrame.from_dict(_lane_ids__mean_speed_over_time).sort_index()
        df__vehicle_ids__normalized_speed_limits_over_time = pd.DataFrame.from_dict(_vehicle_ids__normalized_speed_limit_over_time).sort_index()  # normalized by the current lane limit
        df__vehicle_ids__driven_meters_over_time = pd.DataFrame.from_dict(_vehicle_ids__driven_meters_over_time).sort_index()
        df__vehicle_ids__current_speed_over_time = pd.DataFrame.from_dict(_vehicle_ids__current_speed_over_time).sort_index()
        n_controlled_lanes, n_controlled_tls = len(_lane_ids__actions_over_time), len(_traffic_light_ids__actions_over_time)

        physical_speed_prediction_measure: Optional[str] = None
        if is_speed_advice is True:
            if _gym_env.config.lane_config is not None:
                physical_speed_prediction_measure = _gym_env.config.lane_config.physical_prediction_measure
            elif _gym_env.config.lane_compound_config is not None:
                physical_speed_prediction_measure = _gym_env.config.lane_compound_config.physical_prediction_measure

        print()
        print(episode_end_metrics)

    # Write local variables to cache file. Leave out variables with leading underscore.
    _allowed_types = (bool, int, float, str, dict, list, tuple, pd.DataFrame, pd.Series)
    _local_vars_dict = {k: v for k, v in locals().items() \
                        if not k.startswith("_") and isinstance(v, _allowed_types) and k not in ("In", "Out")}
    with open(CACHE_FILE_PATH, mode="wb") as _file:
        pickle.dump(_local_vars_dict, file=_file)


def test_plot_route_progress():
    assert CACHE_FILE_PATH.is_file(), f"Collected data cache '{CACHE_FILE_PATH.name}' does not exist. Perhaps you " \
                                      f"need to run '{test_generate_inference_statistics_cache.__name__}()' first!"
    with open(CACHE_FILE_PATH, mode="rb") as file:
        data: dict[str, Any] = pickle.load(file)
    _plot_route_progress(time_interval_seconds=range(0, 200), figsize=(12, 6.5), filter_lane_ids="w_t_0", **data)


class _ExpNorm(matplotlib.colors.FuncNorm):
    """Normalizes using an exponential function y = (a**x - 1) / (a - 1),  which maps from [0; 1] to [0; 1]."""
    def __init__(self, a: int = 5, vmin: Optional[float] = None, vmax: Optional[float] = None, clip: bool = False):
        self._a = a
        super().__init__((self._map_forward, self._map_inverse), vmin=vmin, vmax=vmax, clip=clip)

    def _map_forward(self, x):
        return (self._a**x - 1) / (self._a - 1)

    def _map_inverse(self, y):
        _inner = y*(self._a-1) + 1
        return np.log(_inner) / np.log(self._a)


def _plot_route_progress(
        is_speed_advice: bool,
        trip_info_objects: list[TripInfo],
        vehicle_ids__route_ids: dict[str, str],
        lane_ids__vehicle_ids: dict[str, set[str]],
        df__vehicle_ids__driven_meters_over_time: pd.DataFrame,
        df__vehicle_ids__normalized_speed_limits_over_time: pd.DataFrame,
        filter_lane_ids: Optional[Union[str, List[str]]] = None,  # If desired, filter plotted vehicles by affiliation to lane id(s)
        time_interval_seconds: range = range(0, 100),
        figsize: Tuple[float, float] = (12, 6.5),
        upsample_factor: int = 20,
        **kwargs):
    if isinstance(filter_lane_ids, str):
        filter_lane_ids = [filter_lane_ids]

    # Prepare upsampling of the time-based data, in order to close potential gaps in scatter plots
    if is_speed_advice:
        assert len(df__vehicle_ids__driven_meters_over_time) == len(df__vehicle_ids__normalized_speed_limits_over_time)
        pd.testing.assert_index_equal(df__vehicle_ids__driven_meters_over_time.index, df__vehicle_ids__normalized_speed_limits_over_time.index)
    _original_index = np.arange(df__vehicle_ids__driven_meters_over_time.index.min(), df__vehicle_ids__driven_meters_over_time.index.max()+1)
    _upsampled_index = np.arange(_original_index[0], _original_index[-1]+1, 1 / upsample_factor)

    # Prepare the color map
    # _color_map = cm.get_cmap("winter")
    _color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "pink", "red", "orange", "lightgreen"])
    _color_normalizer = matplotlib.colors.Normalize(vmax=df__vehicle_ids__normalized_speed_limits_over_time.min().min(), vmin=df__vehicle_ids__normalized_speed_limits_over_time.max().max(), clip=True)
    # _color_normalizer = matplotlib.colors.Normalize(df__vehicle_ids__normalized_speed_limits_over_time.min().min(), df__vehicle_ids__normalized_speed_limits_over_time.max().max(), clip=True)
    _color_normalizer = _ExpNorm(a=20, vmin=df__vehicle_ids__normalized_speed_limits_over_time.min().min(), vmax=df__vehicle_ids__normalized_speed_limits_over_time.max().max(), clip=True)

    route_ids__vehicle_ids: dict[str, set[str]] = {r: {v for v in vehicle_ids__route_ids.keys() if vehicle_ids__route_ids[v] == r} for r in vehicle_ids__route_ids.values()}
    vehicle_ids__trip_info: dict[str, TripInfo] = {t.id: t for t in trip_info_objects}

    _set_fig_style()
    plt.figure(figsize=figsize, dpi=300)
    for _vehicle_id in vehicle_ids__trip_info.keys():
        if filter_lane_ids is not None and not all(_vehicle_id in lane_ids__vehicle_ids[l] for l in filter_lane_ids):
            continue
        if _vehicle_id not in vehicle_ids__trip_info:
            continue
        _trip_info = vehicle_ids__trip_info[_vehicle_id]
        if _trip_info.depart not in time_interval_seconds:
            continue
        _relative_progress_over_time: pd.Series = df__vehicle_ids__driven_meters_over_time[_vehicle_id] / _trip_info.routeLength

        if is_speed_advice:
            _normalized_speed_limits_over_time: pd.Series = df__vehicle_ids__normalized_speed_limits_over_time[_vehicle_id]
            c = np.interp(_upsampled_index, _original_index, _normalized_speed_limits_over_time.values)
        else:
            c = "lightgreen"

        scatter = plt.scatter(
            x=_upsampled_index, s=0.5, cmap=_color_map, norm=_color_normalizer,
            y=np.interp(_upsampled_index, df__vehicle_ids__driven_meters_over_time.index, _relative_progress_over_time.values),
            c=c)

    if is_speed_advice:
        cbar = plt.colorbar(scatter)
        cbar.ax.set_ylabel("Lane speed limit\n(norm. to maximum lane speed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Vehicles route progress")
    plt.tight_layout()
    plt.show()
    # plt.savefig(Path("~/Desktop/ExperimentsAndResults-RouteProgress-Grid4x4NoAnnotations.pdf").expanduser())


def test_plot_action_distributions__all_actuators():
    assert CACHE_FILE_PATH.is_file(), f"Collected data cache '{CACHE_FILE_PATH.name}' does not exist. Perhaps you " \
                                      f"need to run '{test_generate_inference_statistics_cache.__name__}()' first!"
    with open(CACHE_FILE_PATH, mode="rb") as file:
        data: dict[str, Any] = pickle.load(file)
    _plot_action_distributions__all_actuators(**data, lane_actions_ewm_halflife=None)


def _plot_action_distributions__all_actuators(
        net_file_stem: str, physical_speed_prediction_measure: Optional[str],
        df__traffic_light_ids__actions_over_time: pd.DataFrame, df__lane_ids__actions_over_time: pd.DataFrame,
        df__lane_ids__normalized_speed_limits_over_time: pd.DataFrame,
        lane_actions_ewm_halflife: Optional[int] = None,
        **kwargs):

    _ewm_filter_fn = (lambda x: x) if lane_actions_ewm_halflife is None \
        else (lambda x: x.ewm(halflife=lane_actions_ewm_halflife).mean())
    _n_controlled_lanes, _n_controlled_tls = len(df__lane_ids__actions_over_time.columns), \
        len(df__traffic_light_ids__actions_over_time.columns)
    _actuator_ids = list(df__traffic_light_ids__actions_over_time.columns) + list(df__lane_ids__actions_over_time.columns)
    _actuator_types: list[str] = ["TrafficLight"] * _n_controlled_tls + ["Lane"] * _n_controlled_lanes
    _action_series_list: list[pd.Series] = \
        [df__traffic_light_ids__actions_over_time[id] for id in df__traffic_light_ids__actions_over_time.columns] + \
        [df__lane_ids__actions_over_time[id] for id in df__lane_ids__actions_over_time.columns]

    _set_fig_style()
    n_cols = 4 if _n_controlled_lanes > 0 else 2
    n_rows = len(_action_series_list)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 2.5*n_rows))
    fig.suptitle(net_file_stem + (f" -- lanes predicting {physical_speed_prediction_measure}" if _n_controlled_lanes > 0 else ""))
    for _actuator_id, _actuator_type, _action_series, _ax in zip(_actuator_ids, _actuator_types, _action_series_list, axes):
        _action_series = _ewm_filter_fn(_action_series) if _actuator_type == "Lane" else _action_series
        _ax[0].set_ylabel(f"{_actuator_type}[{_actuator_id}]", rotation=0, size="large")
        _ax[0].set_title("Actions distribution")
        _ax[0].hist(_action_series.values, bins=2 if _actuator_type == "TrafficLight" else 50)
        _ax[1].set_title("Actions over time")
        _ax[1].plot(_action_series)
        if _actuator_type == "Lane":
            _ax[0].set_xlim(-0.1, 0.1)
            _ax[1].set_ylim(-0.1, 0.1)
            _speed_limits_over_time = _ewm_filter_fn(df__lane_ids__normalized_speed_limits_over_time[_actuator_id])
            _ax[2].set_title("Speed limit (normalized) distribution")
            _ax[2].hist(_speed_limits_over_time.values, bins=50)
            _ax[3].set_title("Speed limits (normalized) over time")
            _ax[3].plot(_speed_limits_over_time)
        else:
            _ax[2].set_axis_off()
            _ax[3].set_axis_off()
    fig.tight_layout()
    fig.show()


def test_plot_action_distributions__single_lane():
    assert CACHE_FILE_PATH.is_file(), f"Collected data cache '{CACHE_FILE_PATH.name}' does not exist. Perhaps you " \
                                      f"need to run '{test_generate_inference_statistics_cache.__name__}()' first!"
    with open(CACHE_FILE_PATH, mode="rb") as file:
        data: dict[str, Any] = pickle.load(file)
    _plot_action_distributions__single_lane(**data, lane_id="n_t_1")


def _plot_action_distributions__single_lane(
        df__lane_ids__actions_over_time: pd.DataFrame,
        df__lane_ids__normalized_speed_limits_over_time: pd.DataFrame,
        lane_id: str,
        **kwargs):

    assert lane_id in df__lane_ids__actions_over_time.columns, f"Lane '{lane_id}' does not exist!"

    _actions_series = df__lane_ids__actions_over_time[lane_id]
    _speed_limits_series = df__lane_ids__normalized_speed_limits_over_time[lane_id]
    _first_color, _second_color = next(colors), next(colors)

    _set_fig_style()

    def _plot_distribution_and_timeseries(_series: pd.Series, ylabel: str, marker_alpha: float):
        fig, axes = plt.subplot_mosaic("AABBBBBBB", sharey=True, figsize=(7, 4), dpi=300)
        axes["A"].hist(_series.values, bins=50, alpha=0.5, orientation="horizontal")
        # axes["A"].set_ylim(-0.1, 0.1)
        axes["A"].set_ylabel(ylabel)
        _smoothed_actions = _series.rolling(window=5).median()
        axes["B"].scatter(x=_smoothed_actions.index, y=_smoothed_actions.values, color=_first_color,
                          alpha=marker_alpha, label="Raw signal", s=10)
        axes["B"].plot(_series.ewm(halflife=40).mean().iloc[40:], color=_second_color, label="Smoothed")
        axes["B"].set_xlabel("Time [s]")
        legend = axes["B"].legend()
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        fig.tight_layout()
        return fig

    fig = _plot_distribution_and_timeseries(_actions_series, ylabel="Predicted speed change", marker_alpha=0.2)
    fig.show()
    # fig.savefig(Path("~/Desktop/ExperimentsAndResults-DistributionPlots-SpeedChange.pdf").expanduser())

    fig = _plot_distribution_and_timeseries(_speed_limits_series, ylabel="Resulting speed limit\n(norm. to lane speed limit)", marker_alpha=0.1)
    fig.show()
    # fig.savefig(Path("~/Desktop/ExperimentsAndResults-DistributionPlots-SpeedLimit.pdf").expanduser())

"""This script helps to evaluate trained models. Results are put into training-compound dirs as YAML files."""
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from util.finished_trainings.evaluation import run_multiple_inferences, \
    FILENAME__EVALUATION_RUN_RESULTS_TABLE, FILENAME__EVALUATION_RESULTS_SUMMARY

from util.finished_trainings.path_lookup import get_single_trial_dirs


OUTPUTS_DIR = Path(__file__).parent / "outputs"
N_EVALUATION_RUNS = 20  # Perform this number of eval runs on each trained model, and determine µ and std


def _delete_evaluation_outputs():
    """Helper function that deletes all evaluation outputs written so far. May be called from below main, if desired!"""
    _trial_compound_folders = list(p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and "trial-compound" in p.name and "training-si" in p.name)
    _trial_compound_folders = [p for p in _trial_compound_folders]
    for _trial_compound_dir in _trial_compound_folders:
        # Iterate over each single trial
        for _trial_dir in get_single_trial_dirs(trial_compound_dir=_trial_compound_dir):
            _results_table_path = _trial_dir / FILENAME__EVALUATION_RUN_RESULTS_TABLE
            if _results_table_path.is_file():
                _results_table_path.unlink()

        _results_summary_path = _trial_compound_dir / FILENAME__EVALUATION_RESULTS_SUMMARY
        if _results_summary_path.is_file():
            _results_summary_path.unlink()


if __name__ == '__main__':
    assert OUTPUTS_DIR.is_dir(), f"Outputs dir '{OUTPUTS_DIR}' does not exist!"

    # _delete_evaluation_outputs()    --> Handle with caution!
    _trial_compound_folders = list(p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and "trial-compound" in p.name)
    # TODO: Insert training filter operation here, e.g.
    #       _trial_compound_folders = [p for p in _trial_compound_folders if "my_cool_training" p.name]
    for _trial_compound_dir in tqdm(_trial_compound_folders):
        print(f"Start taking a look at trial-compound '{_trial_compound_dir.name}'")
        # Iterate over each single trial
        compound_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for _trial_dir in get_single_trial_dirs(trial_compound_dir=_trial_compound_dir):
            _results_table_path = _trial_dir / FILENAME__EVALUATION_RUN_RESULTS_TABLE
            if _results_table_path.is_file():
                _df_episode_end_metrics = \
                    pd.read_csv(_results_table_path, sep="\t", encoding="utf-8", index_col=None, header=0)
            else:
                print(f"Running for '{_trial_dir.name}'")
                _df_episode_end_metrics = run_multiple_inferences(_trial_dir, n_runs=N_EVALUATION_RUNS)
                _df_episode_end_metrics.to_csv(
                    _results_table_path, sep="\t", encoding="utf-8", index=False, header=True)
            for _metric in ("resco_delay", "emissions/CO2", "n_finished_trips"):
                _mu, _std = _df_episode_end_metrics[_metric].mean(), _df_episode_end_metrics[_metric].std()
                compound_scores[_trial_dir.name][_metric] = {"mean": _mu, "std": _std, "std-%": f"{_std / _mu:.2%}"}

        # Compile results & write to summary YAML
        sorted_result_scores = \
            {k: v for k, v in sorted(compound_scores.items(), key=lambda item: item[1]["resco_delay"]["mean"])}
        with open(_trial_compound_dir / FILENAME__EVALUATION_RESULTS_SUMMARY, mode="w", encoding="utf-8") as _file:
            yaml.dump(sorted_result_scores, stream=_file, sort_keys=False)

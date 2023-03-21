"""This script offers utils to lookup paths in training output folders, such as trial-compounds / single-trials."""
import re
from pathlib import Path

_SINGLE_TRIAL_FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}__[a-z0-9]{8}__single-trial.*",
                                          flags=re.IGNORECASE)


def get_single_trial_dirs(trial_compound_dir: Path) -> list[Path]:
    _single_trial_paths = [f for f in trial_compound_dir.iterdir()
                           if f.is_dir() and _SINGLE_TRIAL_FOLDER_PATTERN.match(f.name) is not None]
    return _single_trial_paths

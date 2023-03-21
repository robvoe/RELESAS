import enum
from pathlib import Path, PurePath
from abc import ABC
from dataclasses import dataclass, is_dataclass
from typing import Dict, Any

import dacite
import yaml


@dataclass
class BaseConfig(ABC):
    def save_to_yaml(self, yaml_path: Path):
        with open(yaml_path, mode="w", encoding="utf-8") as f:
            yaml.dump(self._to_yaml_compatible_dict(), stream=f, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "BaseConfig":
        with open(yaml_path, mode="r", encoding="utf-8") as f:
            dict_ = yaml.load(f, Loader=yaml.Loader)
        try:
            instance = cls._from_yaml_compatible_dict(dict_)
            return instance
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            # Provide some information to the user, so that he's not overwhelmed by some cryptic error message
            raise RuntimeError("An error occurred when parsing a YAML file. Invalid YAML?") from e

    def _to_yaml_compatible_dict(self) -> Dict[str, Any]:
        config_dict = {}
        for key, val in self.__dict__.items():
            if isinstance(val, BaseConfig):
                val = val._to_yaml_compatible_dict()
            elif is_dataclass(val):
                raise TypeError(f"Dataclasses which don't inherit from {BaseConfig.__name__} are not allowed in "
                                f"configs. Otherwise, we could end-up with ugly class names after marshalling to "
                                f"YAML.")
            elif isinstance(val, tuple):
                raise TypeError("Tuples are not available in configs, b/c otherwise we could get in trouble with "
                                "unmarshalling from YAML")
            elif isinstance(val, PurePath):
                val = str(val)
            config_dict[key] = val
        return config_dict

    @classmethod
    def _from_yaml_compatible_dict(cls, dict_: Dict[str, Any]) -> "BaseConfig":
        instance = dacite.from_dict(data_class=cls, data=dict_, config=dacite.Config(
            type_hooks={Path: Path},
            cast=[enum.Enum]
        ))
        return instance

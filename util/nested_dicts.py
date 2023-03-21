from typing import Any, Union


def flatten_nested_dict(nested_dict: dict[str, Union[Any, dict[str, Any]]]) -> dict[str, Any]:
    """Unpacks nested dicts,  e.g. {"emissions": {"a": 2}} --> {"emissions/a": 2}"""
    _unpacked = {}
    for _name, _value in nested_dict.items():
        if isinstance(_value, dict):
            _unpacked.update({f"{_name}/{_sub_name}": _sub_val for _sub_name, _sub_val in _value.items()})
        else:
            _unpacked[_name] = _value
    return _unpacked

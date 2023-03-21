from pathlib import Path
from typing import List, Dict, Optional
import pytest

import numpy as np

from .trip_info import TripInfo, from_xml_file
from .route_file_parser import get_vehicle_departs


_VEHICLE_DEPARTS_CACHE: Dict[Path, Dict[str, float]] = {}


def get_median_emissions(trip_info_objects: List[TripInfo]) -> Dict[str, float]:
    """
    Returns a dictionary of median emission scores. This function is intended to be obtained once when an episode ends.

    Remark:
    I intentionally decided against returning both mean and median (see commented code below), because the regarding
    prefixes would greatly confuse users that look at e.g. Tensorboard outputs.  Instead, median must suffice here :-)

    @param trip_info_objects: The parsed TripInfo objects, whose original data was captured from SUMO.
    """
    assert len(trip_info_objects) > 0
    _emission_names: List[str] = list(trip_info_objects[0].emissions_dict.keys())
    _trip_emission_dicts: List[Dict[str, float]] = [t.emissions_dict for t in trip_info_objects]
    # median_emissions: Dict[str, float] = {}
    # for _name in _emission_names:
    #     _scores: List[float] = [e[_name] for e in _trip_emission_dicts]
    #     median_emissions[f"mean_{_name}"] = float(np.mean(_scores))
    #     median_emissions[f"median_{_name}"] = float(np.median(_scores))
    median_emissions = {n: float(np.median([t_em[n] for t_em in _trip_emission_dicts])) for n in _emission_names}
    return median_emissions


def get_resco_episode_delay(trip_info_objects: List[TripInfo], route_file_path: Optional[Path],
                            simulation_end_time: Optional[float]) -> float:
    """
    Determines the episode-wise "Delay (s)" metric á la RESCO (see https://github.com/Pi-Star-Lab/RESCO); see RESCO's
    readXML.py.

    @param trip_info_objects: The parsed TripInfo objects, whose original data was captured from SUMO.
    @param route_file_path: Optional path to the used route file (*.rou.xml). Must be provided if also non-departed
                            vehicles should be considered.
    @param simulation_end_time: Optionally provided time at which an episode would be considered "done". Must be
                                provided if also non-departed vehicles should be considered.
    """
    assert sum(p is None for p in (route_file_path, simulation_end_time)) in (0, 2), \
        "Either both parameters 'route_file_path' and 'simulation_end_time' must be provided, or none of them!"

    _n_trips = len(trip_info_objects)
    _total = sum(trip.timeLoss+trip.departDelay for trip in trip_info_objects)

    # Now let's also consider those vehicles that didn't depart
    if route_file_path is not None:
        if route_file_path not in _VEHICLE_DEPARTS_CACHE:
            _VEHICLE_DEPARTS_CACHE[route_file_path] = get_vehicle_departs(route_file_path=route_file_path)
        _vehicle_departs = _VEHICLE_DEPARTS_CACHE[route_file_path]

        _max_depart_time = max(trip.depart for trip in trip_info_objects)
        _max_depart_time_id = next(filter(lambda t: t.depart == _max_depart_time, trip_info_objects)).id
        if all(t.id in _vehicle_departs.keys() for t in trip_info_objects):
            _max_depart_time = _vehicle_departs[_max_depart_time_id]
            _never_started_vehicle_departs: List[float] = \
                [vd[1] for vd in filter(lambda t: t[1] > _max_depart_time, _vehicle_departs.items())]
            _delay_sum = sum(simulation_end_time-d for d in _never_started_vehicle_departs)
            _total += _delay_sum
            _n_trips += len(_never_started_vehicle_departs)

    return _total / _n_trips


@pytest.fixture
def _trip_info_objects() -> List[TripInfo]:
    trip_info_xml_file_path = Path(__file__).parent.parent / "unit_test_data" / "grid4x4_tripinfo_1.xml"
    trip_info_objects = from_xml_file(trip_info_xml_file_path)
    assert len(trip_info_objects) == 319
    return trip_info_objects


def test_get_resco_episode_delay(_trip_info_objects):
    route_file_path = Path(__file__).parent.parent / "unit_test_data" / "grid4x4_1.rou.xml"
    trip_info_objects = _trip_info_objects

    episode_delay = get_resco_episode_delay(
        trip_info_objects=trip_info_objects, route_file_path=route_file_path, simulation_end_time=3600)
    np.testing.assert_allclose(episode_delay, 31.8544, rtol=1e-3, atol=1e-3)


def test_get_emission_scores(_trip_info_objects):
    trip_info_objects = _trip_info_objects
    emission_scores = get_median_emissions(trip_info_objects=trip_info_objects)
    assert all(emission_name in emission_scores for emission_name in trip_info_objects[0].emissions_dict.keys())

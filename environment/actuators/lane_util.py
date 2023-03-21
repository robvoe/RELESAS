from collections import defaultdict
from typing import Optional, Tuple

import traci


_CACHE__LANE_IDS__SUCCESSOR_IDS: dict[str, dict[str, list[str]]] = {}
_CACHE__LANES_WHICH_ARE_IN_LANES_TO_ANY_TL: dict[str, set[str]] = {}

_CACHE__LANE_IDS__FOLLOWING_TL_IDS: dict[str, dict[str, str]] = {}

SHORT_LANE_THRESHOLD = 15  # Length threshold (in m) below which we consider a lane being short.


def get_lane_leader(sumo_connection: traci.Connection, lane_id: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Returns a lane's leading vehicle ID (standing vehicles are neglected), and -if present- its distance to the next
    standing vehicle. Will be None if there are no vehicles present on the given lane.
    """
    _vehicle_ids = sumo_connection.lane.getLastStepVehicleIDs(lane_id)
    if len(_vehicle_ids) == 0:
        return None, None
    _vehicle_id: str = _vehicle_ids[0]
    while True:
        _leader = sumo_connection.vehicle.getLeader(_vehicle_id)
        if _leader is None or sumo_connection.vehicle.getLaneID(_leader[0]) != lane_id:
            return _vehicle_id, None
        if sumo_connection.vehicle.getWaitingTime(_leader[0]) > 0:
            _distance_to_next_standing_vehicle = _leader[1]
            return _vehicle_id, _distance_to_next_standing_vehicle
        _vehicle_id = _leader[0]


def get_next_traffic_light_stats(sumo_connection: traci.Connection, vehicle_id: str) -> Optional[Tuple]:
    """
    Returns a few statistics about a vehicle's upcoming traffic light. These are namely distance, its state and id.
    Will be None in case there's no traffic light ahead of the vehicle.
    """
    _upcoming_traffic_lights = sumo_connection.vehicle.getNextTLS(vehicle_id)
    if len(_upcoming_traffic_lights) == 0:
        return None
    _sorted = sorted(_upcoming_traffic_lights, key=lambda s: s[2], reverse=False)
    traffic_light_id, _, distance, state = _sorted[0]
    return distance, state, traffic_light_id


def get_short_lanes(sumo_connection: traci.Connection) -> list[str]:
    """Provides an ID list of all lanes that can be considered "short"."""
    _controlled_lane_ids = \
        [sumo_connection.trafficlight.getControlledLanes(tl) for tl in sumo_connection.trafficlight.getIDList()]
    _controlled_lane_ids = set(lane_id for sublist in _controlled_lane_ids for lane_id in sublist)
    _lane_ids__lengths = {lane: sumo_connection.lane.getLength(lane) for lane in _controlled_lane_ids}

    _short_lanes = [lane for lane, length in _lane_ids__lengths.items() if length < SHORT_LANE_THRESHOLD]
    return _short_lanes


def get_inflowing_lanes(sumo_connection: traci.Connection, lane_id: str) -> list[str]:
    """
    Determines a list of lanes which lead/flow into a given lane. This function is likely to be used in conjunction
    with "short" lanes, in order to extend their (very limited) observation spaces.
    """
    # Cache away SUMO communication-heavy payloads
    _scenario_specific_cache_key = sumo_connection.simulation.getNetBoundary()  # TODO Net file name would be better..
    if _scenario_specific_cache_key not in _CACHE__LANE_IDS__SUCCESSOR_IDS:
        _lane_ids__successor_ids = \
            {lane: [link[0] for link in sumo_connection.lane.getLinks(lane, extended=False)]
             for lane in sumo_connection.lane.getIDList() if not lane.startswith(":")}
        _CACHE__LANE_IDS__SUCCESSOR_IDS[_scenario_specific_cache_key] = _lane_ids__successor_ids
    if _scenario_specific_cache_key not in _CACHE__LANES_WHICH_ARE_IN_LANES_TO_ANY_TL:
        _tl_id__in_lane_lists = {tl_id: [lane for lane in sumo_connection.trafficlight.getControlledLanes(tl_id)]
                                 for tl_id in sumo_connection.trafficlight.getIDList()}
        _lanes_which_are_in_lanes_to_tl = set(lane for sublist in _tl_id__in_lane_lists.values() for lane in sublist)
        _CACHE__LANES_WHICH_ARE_IN_LANES_TO_ANY_TL[_scenario_specific_cache_key] = _lanes_which_are_in_lanes_to_tl

    _lane_ids__successor_ids = _CACHE__LANE_IDS__SUCCESSOR_IDS[_scenario_specific_cache_key]
    _lanes_which_are_in_lanes_to_tl = _CACHE__LANES_WHICH_ARE_IN_LANES_TO_ANY_TL[_scenario_specific_cache_key]

    # Find lane-specific predecessors. Remove those predecessors, which are in-lanes to traffic lights
    _unfiltered_predecessors_of_lane = [l for l, succ_list in _lane_ids__successor_ids.items() if lane_id in succ_list]
    _valid_predecessors_of_lane = [l for l in _unfiltered_predecessors_of_lane if l not in _lanes_which_are_in_lanes_to_tl]
    return _valid_predecessors_of_lane


def get_speed_controllable_lanes(sumo_connection: traci.Connection, filter_edge_id: Optional[str],
                                 discard_short_lanes: bool = True) -> list[str]:
    """
    Returns lanes that make sense to be speed-controlled in a given scenario.

    @param sumo_connection: Connection instance to SUMO
    @param filter_edge_id: If None, all controllable lanes from the scenario will be provided. If not None, only lanes
                           belonging to the specified edge (i.e. street) are considered.
    @param discard_short_lanes: If True, short lanes will be discarded.
    """
    _lane_ids: list[str] = sumo_connection.lane.getIDList()
    # Retrieve the controlled lanes of all traffic lights. That's necessary for filtering below
    _controlled_lane_ids = \
        [sumo_connection.trafficlight.getControlledLanes(tl) for tl in sumo_connection.trafficlight.getIDList()]
    _controlled_lane_ids = set(lane_id for sublist in _controlled_lane_ids for lane_id in sublist)
    # Apply some filtering
    _lane_ids = [_id for _id in _lane_ids if not _id.strip().startswith(":")]
    _lane_ids = [_id for _id in _lane_ids if _id in _controlled_lane_ids]  # Filter those, which do not lead to a TL
    if discard_short_lanes is True:
        _lane_ids = [l for l in _lane_ids if sumo_connection.lane.getLength(l) >= SHORT_LANE_THRESHOLD]  # Filter out short lanes
    if filter_edge_id is not None:
        _lane_ids = [_id for _id in _lane_ids if sumo_connection.lane.getEdgeID(_id) == filter_edge_id]
    _no_pedestrians = [_id for _id in _lane_ids if "pedestrian" not in sumo_connection.lane.getAllowed(_id)]
    if len(_no_pedestrians) > 0:
        _lane_ids = _no_pedestrians  # In some erroneous maps (e.g. cologne8), pedestrians are allowed on every lane
    return _lane_ids


def get_lane_specific_traffic_light(sumo_connection: traci.Connection, lane_ids: list[str]) -> dict[str, Optional[str]]:
    """
    With a given list of lane ids, this function determines the according (directly following) traffic light ids. If
    a lane has no following traffic light, its individual return value is None.

    @param sumo_connection: Connection to SUMO
    @param lane_ids: List of lane ids for which we want to determine (directly following) traffic light ids.

    @return: Dict of lane ids, mapping to their respective traffic light ids. If a lane has no TL, it maps to None.
    """
    _scenario_specific_cache_key = sumo_connection.simulation.getNetBoundary()  # TODO Net file name would be better..
    if _scenario_specific_cache_key not in _CACHE__LANE_IDS__FOLLOWING_TL_IDS:
        _global_lane_ids__tl_ids = {l: None for l in sumo_connection.lane.getIDList()}
        for _tl_id in sumo_connection.trafficlight.getIDList():
            _controlled_lanes = sumo_connection.trafficlight.getControlledLanes(_tl_id)
            _global_lane_ids__tl_ids.update({l: _tl_id for l in _controlled_lanes})
        _CACHE__LANE_IDS__FOLLOWING_TL_IDS[_scenario_specific_cache_key] = _global_lane_ids__tl_ids

    _global_lane_ids__tl_ids = _CACHE__LANE_IDS__FOLLOWING_TL_IDS[_scenario_specific_cache_key]

    assert all(l in _global_lane_ids__tl_ids for l in lane_ids), "At least one of the passed lane ids does not " \
                                                                 "appear in the list of global lanes"
    lane_ids__tl_ids = {l: _global_lane_ids__tl_ids[l] for l in lane_ids}
    return lane_ids__tl_ids


def cluster_lanes_by_edge(sumo_connection: traci.Connection, lane_ids: list[str]) -> dict[str, list[str]]:
    """
    Clusters a list of lanes by their edges.

    @param sumo_connection: Connection to SUMO
    @param lane_ids: List of lane ids which we want to cluster.
    @return: Dict of edge ids, mapping to lists of lane ids.
    """
    _edge_ids__lane_ids = defaultdict(list)
    for _lane_id in lane_ids:
        _edge_id = sumo_connection.lane.getEdgeID(_lane_id)
        _edge_ids__lane_ids[_edge_id] += [_lane_id]
    return _edge_ids__lane_ids


def cluster_lanes_by_traffic_light(sumo_connection: traci.Connection, lane_ids: list[str]) -> dict[str, list[str]]:
    """
    Clusters a list of lanes by the traffic lights they lead to. Lanes leading to no TL are discarded.

    @param sumo_connection: Connection to SUMO
    @param lane_ids: List of lane ids which we want to cluster.
    @return: Dict of TL ids, mapping to lists of lane ids.
    """
    _lane_ids__tl_ids = get_lane_specific_traffic_light(sumo_connection, lane_ids=lane_ids)

    _tl_ids__lane_ids = {}
    for _tl_id in set(_lane_ids__tl_ids.values()):
        if _tl_id is None:
            continue
        _lanes = [l for l, tl in _lane_ids__tl_ids.items() if tl == _tl_id]
        assert len(_lanes) > 0
        _tl_ids__lane_ids[_tl_id] = _lanes
    return _tl_ids__lane_ids

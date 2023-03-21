from copy import deepcopy
from pathlib import Path
from typing import Tuple, Optional, Union, Any

import traci
from gym.core import ActType, ObsType
import yaml

from ..base_env import BaseGymEnv


class _EdgeUsageEnv(BaseGymEnv):
    """An env that counts edge usages by switching all TLs green, while having vehicle collisions disabled."""
    def __init__(self, config: BaseGymEnv.Config, simulation_end_time: float):
        config = deepcopy(config)  # We don't want to touch the original config object
        config.additional_sumo_args = ["--collision.action", "none",
                                       "--collision.mingap-factor", "0",
                                       "--collision.check-junctions", "false",
                                       "--no-warnings", "true"]
        config.do_process_trip_info = False

        super().__init__(config=config)  # Starts SUMO
        self._set_all_traffic_lights_green()

        self._simulation_end_time = simulation_end_time
        self._known_vehicle_ids: set[str] = set()
        self.edge_ids__vehicle_ids: dict[str, set[str]] = \
            {edge: set() for edge in self._sumo_connection.edge.getIDList() if not edge.startswith(":")}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,
              **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset()
        return "not-a-valid-return-value-but-that's-okay-here"

    def render(self, mode="human"):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self._sumo_connection.simulationStep()

        # Remove right-of-way (and other) checks from newly spawned vehicles
        # (--> see https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html)
        for _vehicle_id in self._sumo_connection.vehicle.getIDList():
            if _vehicle_id not in self._known_vehicle_ids:
                self._known_vehicle_ids.add(_vehicle_id)
                self._sumo_connection.vehicle.setSpeedMode(_vehicle_id, 32)

        # Look which vehicles drive on which edges
        for _edge_id, _vehicle_ids in self.edge_ids__vehicle_ids.items():
            _current_vehicle_ids = self._sumo_connection.edge.getLastStepVehicleIDs(_edge_id)
            _vehicle_ids.update(_current_vehicle_ids)

        done = self._sumo_connection.simulation.getTime() >= self._simulation_end_time or \
               self._sumo_connection.simulation.getMinExpectedNumber() == 0
        return "not-a-valid-return-value", "yet-another-invalid-value", done, "hello-invalid-value"

    def _set_all_traffic_lights_green(self) -> None:
        _connection = self._sumo_connection
        for _tl_id in _connection.trafficlight.getIDList():
            _logic: traci.trafficlight.Logic = _connection.trafficlight.getAllProgramLogics(_tl_id)[0]
            _phases: list[traci.trafficlight.Phase] = list(_logic.phases)
            assert all(len(p.state) == len(_phases[0].state) for p in _phases), \
                "Phase states are expected to be of equal length!"

            _all_green_state = "G" * len(_phases[0].state)
            _logic.type = 0
            _logic.currentPhaseIndex = 0
            _logic.phases = [traci.trafficlight.Phase(duration=20_000, state=_all_green_state)] * len(_phases)
            _connection.trafficlight.setProgramLogic(_tl_id, _logic)
            _connection.trafficlight.setRedYellowGreenState(_tl_id, _all_green_state)

    @property
    def get_simulation_time(self) -> float:
        return self._sumo_connection.simulation.getTime()


def _get_theoretical_edge_usage(base_env_config: BaseGymEnv.Config, simulation_end_time: float) -> dict[str, float]:
    """Internal function that does all the work"""
    _env = _EdgeUsageEnv(config=base_env_config, simulation_end_time=simulation_end_time)
    _env.reset()
    _done = False
    _time__n_vehicles = {}
    while _done is False:
        _, _, _done, _ = _env.step(action="no-valid-action :-)")
        _time__n_vehicles[int(_env.get_simulation_time)] = len(_env._sumo_connection.vehicle.getIDList())
    simulation_seconds = _env.get_simulation_time - base_env_config.simulation_begin_time
    _env.close()
    return {edge_id: len(vehicles)/simulation_seconds*3600 for edge_id, vehicles in _env.edge_ids__vehicle_ids.items()}


def get_theoretical_edge_usage(base_env_config: BaseGymEnv.Config, simulation_end_time: float) -> dict[str, float]:
    """
    Determines, with all traffic lights green & no right-of-way rules, how many vehicles drive on which edges.

    Since this function might run very long, its results are cached next to the to-be-examined route file.

    @param base_env_config: Specifies which scenario we want to examine under which conditions.
    @param simulation_end_time: End time of the simulation, in seconds.
    @return: A dict that tells the edge usage (normalized to vehicles per simulated hour).
    """
    _from, _to = base_env_config.simulation_begin_time, simulation_end_time
    _cache_file_name = f"{base_env_config.route_file_path.stem}.edge-usage.{_from}s-{_to}s.yaml"
    _cache_file_path: Path = base_env_config.route_file_path.parent / _cache_file_name
    if _cache_file_path.is_file():
        with open(_cache_file_path, mode="r", encoding="utf-8") as file:
            _data: dict[Any, float] = yaml.safe_load(file)  # Keys might be interpreted as non-strings. Fix in next line
            return {(edge_id if isinstance(edge_id, str) else str(edge_id)): n_veh for edge_id, n_veh in _data.items()}

    _data = _get_theoretical_edge_usage(base_env_config=base_env_config, simulation_end_time=simulation_end_time)
    try:
        with open(_cache_file_path, mode="w", encoding="utf-8") as file:
            file.write(f"# Cached theoretical edge usage in the timeframe from {_from}s to {_to}s\n")
            file.write(f"# For more info see:  {Path(__file__).name}, {get_theoretical_edge_usage.__name__}()")
            yaml.dump(_data, stream=file, encoding="utf-8", sort_keys=True)
    except IOError:
        pass  # We cannot write to the cache file. This is okay at that point :-)
    return _data


def test_get_edge_usage__dev():
    import matplotlib.pyplot as plt
    from .. import env_config_templates

    env_config = env_config_templates.sumo_rl_single_intersection__high_traffic()
    # env_config.use_gui = True

    # D = get_theoretical_edge_usage(env_config, simulation_end_time=env_config.simulation_end_time)
    D = _get_theoretical_edge_usage(env_config, simulation_end_time=env_config.simulation_end_time)
    print(D)

    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()), rotation=90)
    plt.suptitle(env_config.route_file_stem)
    plt.show()

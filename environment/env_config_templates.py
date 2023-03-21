# This file provides template GymEnv config objects, especially for RESCO environments. For more
# information on RESCO, see https://github.com/Pi-Star-Lab/RESCO
import inspect
import sys
from pathlib import Path as _Path
from typing import Callable

from .env import GymEnv


_SCENARIOS_ROOT_PATH = _Path(__file__).parent.parent / "scenarios"

_SUMO_RL_PATH = _SCENARIOS_ROOT_PATH / "SUMO-RL"
_RESCO_PATH = _SCENARIOS_ROOT_PATH / "RESCO"


def resco_grid4x4() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "grid4x4" / "grid4x4.net.xml",
        route_file_path=_RESCO_PATH / "grid4x4" / "grid4x4_1.rou.xml",
        simulation_end_time=3_600
    )
    return config


def resco_arterial4x4() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "arterial4x4" / "arterial4x4.net.xml",
        route_file_path=_RESCO_PATH / "arterial4x4" / "arterial4x4_1.rou.xml",
        simulation_end_time=3_600
    )
    return config


def resco_cologne1() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "cologne1" / "cologne1.net.xml",
        route_file_path=_RESCO_PATH / "cologne1" / "cologne1.rou.xml",
        simulation_begin_time=25_200,
        simulation_end_time=28_800
    )
    return config


def resco_cologne3() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "cologne3" / "cologne3.net.xml",
        route_file_path=_RESCO_PATH / "cologne3" / "cologne3.rou.xml",
        simulation_begin_time=25_200,
        simulation_end_time=28_800
    )
    return config


def resco_cologne8() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "cologne8" / "cologne8.net.xml",
        route_file_path=_RESCO_PATH / "cologne8" / "cologne8.rou.xml",
        simulation_begin_time=25_200,
        simulation_end_time=28_800
    )
    return config


def resco_ingolstadt1() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "ingolstadt1" / "ingolstadt1.net.xml",
        route_file_path=_RESCO_PATH / "ingolstadt1" / "ingolstadt1.rou.xml",
        simulation_begin_time=57_600,
        simulation_end_time=61_200
    )
    return config


def resco_ingolstadt7() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "ingolstadt7" / "ingolstadt7.net.xml",
        route_file_path=_RESCO_PATH / "ingolstadt7" / "ingolstadt7.rou.xml",
        simulation_begin_time=57_600,
        simulation_end_time=61_200
    )
    return config


def resco_ingolstadt21() -> GymEnv.Config:
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_RESCO_PATH / "ingolstadt21" / "ingolstadt21.net.xml",
        route_file_path=_RESCO_PATH / "ingolstadt21" / "ingolstadt21.rou.xml",
        simulation_begin_time=57_600,
        simulation_end_time=61_200
    )
    return config


def sumo_rl_single_intersection() -> GymEnv.Config:
    """SingleIntersection scenario with ~2500 veh/hour. Vehicle streams are NOT equally distributed!"""
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection.net.xml",
        route_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection.rou.xml",
        simulation_begin_time=0,
        simulation_end_time=3600,
    )
    return config


def sumo_rl_single_intersection__equally_distributed() -> GymEnv.Config:
    """SingleIntersection scenario with ~2500 veh/hour. Vehicle streams are equally distributed!"""
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection.net.xml",
        route_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection-equally-distributed.rou.xml",
        simulation_begin_time=0,
        simulation_end_time=3600,
    )
    return config


def sumo_rl_single_intersection__high_traffic() -> GymEnv.Config:
    """SingleIntersection scenario with ~3200 veh/hour. Vehicle streams are equally distributed!"""
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection.net.xml",
        route_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection-high-traffic.rou.xml",
        simulation_begin_time=0,
        simulation_end_time=3600,
    )
    return config


def sumo_rl_single_intersection__low_traffic() -> GymEnv.Config:
    """SingleIntersection scenario with ~500 veh/hour. Vehicle streams are equally distributed!"""
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection.net.xml",
        route_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection-low-traffic.rou.xml",
        simulation_begin_time=0,
        simulation_end_time=3600,
    )
    return config


def sumo_rl_single_intersection__very_low_traffic() -> GymEnv.Config:
    """SingleIntersection scenario with ~70 veh/hour. Vehicle streams are equally distributed!"""
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection.net.xml",
        route_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection-very-low-traffic.rou.xml",
        simulation_begin_time=0,
        simulation_end_time=3600,
    )
    return config


def sumo_rl_single_intersection__temporary_doubling() -> GymEnv.Config:
    """SingleIntersection scenario with ~1800 veh/hour. During minutes 30..60, flow rate is doubled to ~3600veh/hour"""
    config = GymEnv.Config(
        use_gui=False,
        net_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection-long-roads.net.xml",
        route_file_path=_SUMO_RL_PATH / "single-intersection" / "single-intersection-temporary-doubling.rou.xml",
        simulation_begin_time=0,
        simulation_end_time=7200,  # We give the vehicle stream a bit more time to recover after the inrush
    )
    return config


FUNCTION_NAMES__TEMPLATE_FUNCTIONS: dict[str, Callable[[], GymEnv.Config]] = \
    {n: f for n, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction) if not n.startswith("test_")}

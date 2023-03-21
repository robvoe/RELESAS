from .. import env_config_templates
from ..env import GymEnv
from .lane_util import get_short_lanes


def test_get_short_lanes():
    env_configs__short_lane_id_stems: list[tuple[GymEnv.Config, list[str]]] = [
       (env_config_templates.sumo_rl_single_intersection(), []),
       (env_config_templates.resco_grid4x4(), []),
       (env_config_templates.resco_arterial4x4(), []),
       (env_config_templates.resco_cologne1(), []),
       (env_config_templates.resco_cologne3(), ["200818108#0_1", "200818108#0_0", "319261593#16_0", "319261593#16_1"]),
       (env_config_templates.resco_cologne8(), ["-225249129#0_0"]),
       (env_config_templates.resco_ingolstadt1(), ["164051413"]),
       (env_config_templates.resco_ingolstadt7(), ["10425609#1", "124812856#1", "164051413", "285716192#0.83",
                                                   "-24693977#0", "168702040#4"]),
       (env_config_templates.resco_ingolstadt21(),
        {"-201963533#0", "-18809672#1", "137133006#1", "176550246", "-170018165#1", "170018165#0",
         "612075153#1", "315358250#1", "-447569997#0", "124812856#1", "-4942389#0", "201238726#1",
         "202092676#0", "164051413", "285716192#0.83", "233675413#3", "315358251#1", "168702040#4",
         "128361109#4", "10425609#1", "174800513", "-174800513", "-24693977#0"}),
    ]
    for _env_config, _expected_short_lane_stems in env_configs__short_lane_id_stems:
        # _env_config.use_gui = True  # --> Enable only for debugging purposes!
        _env = GymEnv(config=_env_config, do_output_info=False)
        _short_lanes = get_short_lanes(_env._sumo_connection)
        try:
            assert all(any(lane_id.startswith(stem) for stem in _expected_short_lane_stems) for lane_id in _short_lanes)
            assert all(any(lane_id.startswith(stem) for lane_id in _short_lanes) for stem in _expected_short_lane_stems)
        except AssertionError as e:
            raise RuntimeError(f"Erroneous short-lane-detection for scenario '{_env_config.net_file_stem}'!") from e
        _env.close()

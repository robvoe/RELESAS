import contextlib
from . import env_config_templates
from .env import GymEnv


def test_templates():
    names__functions = env_config_templates.FUNCTION_NAMES__TEMPLATE_FUNCTIONS
    print()
    for (_fn_name, _fn) in names__functions.items():
        if _fn_name.startswith("test_"):
            continue
        print(f"--> Checking '{_fn_name}()'")
        env_config = _fn()
        assert isinstance(env_config, GymEnv.Config)
        assert env_config.net_file_path.is_file()
        assert env_config.route_file_path.is_file()
        with contextlib.closing(GymEnv(env_config, do_output_info=False)) as env:
            action_space = env.action_space
            for _ in range(50):
                env.step(action_space.sample())
        print(f"--> '{_fn_name}()' looks good")
    print("All template envs seem correct")

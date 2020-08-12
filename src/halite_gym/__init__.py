from gym.envs.registration import register
from src.constants import SETTINGS

version_id = "halite-" + SETTINGS["gym"]["version"]

register(
    id=version_id,
    entry_point='gym_foo.envs:FooEnv',
)
from collections.abc import Callable
from functools import partial

from waymax_rl.simulator.env import WaymaxBicycleEnv
from waymax_rl.simulator.observations import obs_global


def create_bicycle_env(
    max_num_objects: int = 64,
    num_envs: int = 1,
    trajectory_length: int = 1,
    observation_fn: Callable | None = obs_global,
    reward_fn: Callable | None = None,
):
    obs_fn = partial(observation_fn, trajectory_length=trajectory_length)

    return WaymaxBicycleEnv(
        max_num_objects=max_num_objects,
        num_envs=num_envs,
        observation_fn=obs_fn,
        reward_fn=reward_fn,
    )

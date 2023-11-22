from collections.abc import Callable
from functools import partial

from waymax_rl.constants import WOD_1_0_0_VALIDATION_BUCKET
from waymax_rl.simulator.env import WaymaxBicycleEnv
from waymax_rl.simulator.observations import obs_vectorize


def create_bicycle_env(
    max_num_objects: int = 64,
    trajectory_length: int = 1,
    observation_fn: Callable | None = obs_vectorize,
    reward_fn: Callable | None = None,
):
    obs_fn = partial(observation_fn, trajectory_length=trajectory_length)

    return WaymaxBicycleEnv(
        max_num_objects=max_num_objects,
        observation_fn=obs_fn,
        reward_fn=reward_fn,
    )


def create_bicycle_env_eval(
    max_num_objects: int = 64,
    num_envs: int = 1,
    path_dataset: str | None = None,
    trajectory_length: int = 1,
    observation_fn: Callable | None = obs_vectorize,
    reward_fn: Callable | None = None,
):
    obs_fn = partial(observation_fn, trajectory_length=trajectory_length)

    if path_dataset is None:
        path_dataset = WOD_1_0_0_VALIDATION_BUCKET

    env = WaymaxBicycleEnv(
        path_dataset=path_dataset,
        max_num_objects=max_num_objects,
        num_envs=num_envs,
        observation_fn=obs_fn,
        reward_fn=reward_fn,
    )

    return env

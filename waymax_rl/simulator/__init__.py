from collections.abc import Callable
from functools import partial

from waymax_rl.simulator.env import WaymaxBicycleEnv
from waymax_rl.simulator.observations import obs_vectorize
from waymax_rl.simulator.rewards import reward_overlap_offroad


def create_bicycle_env(
    max_num_objects: int = 64,
    trajectory_length: int = 1,
    observation_fn: Callable | None = obs_vectorize,
    reward_fn: Callable | None = reward_overlap_offroad,
):
    obs_fn = partial(observation_fn, trajectory_length=trajectory_length)

    return WaymaxBicycleEnv(
        max_num_objects=max_num_objects,
        observation_fn=obs_fn,
        reward_fn=reward_fn,
    )

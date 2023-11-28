from typing import TYPE_CHECKING

import jax


if TYPE_CHECKING:
    from waymax_rl.simulator.env import EnvState, WaymaxBaseEnv


def policy_step(
    env_state: "EnvState",
    env: "WaymaxBaseEnv",
    policy: callable,
    key: jax.random.PRNGKey = None,
) -> tuple:
    # Obtain the current observation from the environment
    observation = env.observe(env_state.simulator_state)

    # Determine actions based on the given policy and observation
    actions = policy(observation, key)

    # Apply the actions to the environment and get the resulting slice of the episode
    env_state, transition = env.step(env_state, actions)

    return env_state, transition


def random_step(
    env: "WaymaxBaseEnv",
    env_state: "EnvState",
    action_shape: tuple,
    key: jax.random.PRNGKey,
    action_bounds: tuple[float, float] = (-1.0, 1.0),
) -> tuple:
    # Generating actions within the specified bounds
    actions = jax.random.uniform(key=key, shape=action_shape, minval=action_bounds[0], maxval=action_bounds[1])

    env_state, transition = env.step(env_state, actions)

    return env_state, transition

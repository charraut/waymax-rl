from typing import TYPE_CHECKING

import jax

from waymax_rl.utils import Transition


if TYPE_CHECKING:
    from waymax.datatypes import SimulatorState

    from waymax_rl.simulator.env import WaymaxBaseEnv


def policy_step(
    env: "WaymaxBaseEnv",
    simulator_state: "SimulatorState",
    policy: callable,
    key: jax.random.PRNGKey = None,
) -> tuple:
    """
    Execute a step in the environment using a given policy.

    Args:
    - env: The environment to interact with.
    - simulator_state: The current state of the simulator.
    - policy: A function that takes an observation and a PRNGKey and returns actions.
    - key: PRNGKey for random number generation.

    Returns:
    - state: The new state of the simulator after the step.
    - transition: A Transition object containing details of the step.
    - metrics: Additional metrics or information returned by the environment after the step.
    """
    # Obtain the current observation from the environment
    obs = env.observe(simulator_state)

    # Determine actions based on the given policy and observation
    actions = policy(obs, key)

    # Apply the actions to the environment and get the resulting slice of the episode
    episode_slice = env.step(simulator_state, actions)

    # Create a transition object to encapsulate the step information
    transition = Transition(
        observation=obs,
        action=actions,
        reward=episode_slice.reward,
        flag=episode_slice.flag,
        next_observation=episode_slice.observation,
        done=episode_slice.done,
    )

    # Extract additional metrics from the episode slice
    metrics = episode_slice.metrics

    return episode_slice.state, transition, metrics


def random_step(
    env: "WaymaxBaseEnv",
    simulator_state: "SimulatorState",
    action_shape: tuple,
    key: jax.random.PRNGKey,
    action_bounds: tuple[float, float] = (-1.0, 1.0),
) -> tuple:
    """
    Perform a random step in the environment.

    Args:
    - env: The environment to interact with.
    - simulator_state: The current state of the simulator.
    - action_shape: The shape of the action space.
    - key: PRNGKey for random number generation.
    - action_bounds: Tuple indicating the lower and upper bounds of the action space.

    Returns:
    - state: The new state of the simulator after the step.
    - transition: A Transition object containing details of the step.
    """
    obs = env.observe(simulator_state)

    # Generating actions within the specified bounds
    actions = jax.random.uniform(key=key, shape=action_shape, minval=action_bounds[0], maxval=action_bounds[1])
    episode_slice = env.step(simulator_state, actions)

    state = episode_slice.state
    transition = Transition(
        observation=obs,
        action=actions,
        reward=episode_slice.reward,
        flag=episode_slice.flag,
        next_observation=episode_slice.observation,
        done=episode_slice.done,
    )

    return state, transition

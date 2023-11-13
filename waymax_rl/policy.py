from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

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
    observation = env.observe(simulator_state)

    # Determine actions based on the given policy and observation
    actions = policy(observation, key)

    # Apply the actions to the environment and get the resulting slice of the episode
    episode_slice = env.step(simulator_state, actions)

    # Create a transition object to encapsulate the step information
    transition = Transition(
        observation=observation,
        action=actions,
        reward=episode_slice.reward,
        flag=episode_slice.flag,
        next_observation=episode_slice.next_observation,
        done=episode_slice.done,
    )

    return episode_slice.next_state, transition


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
    observation = env.observe(simulator_state)

    # Generating actions within the specified bounds
    actions = jax.random.uniform(key=key, shape=action_shape, minval=action_bounds[0], maxval=action_bounds[1])
    episode_slice = env.step(simulator_state, actions)

    transition = Transition(
        observation=observation,
        action=actions,
        reward=episode_slice.reward,
        flag=episode_slice.flag,
        next_observation=episode_slice.next_observation,
        done=episode_slice.done,
    )

    return episode_slice.next_state, transition


def rollout(sim_state: "SimulatorState", env: "WaymaxBaseEnv", policy: callable) -> tuple:
    """Collect trajectories until the environment terminates."""
    init_sim_state, init_transition = policy_step(env, sim_state, policy)
    init_reward = init_transition.reward

    init_carry = (init_sim_state, init_transition, init_reward)

    def cond_fn(carry):
        transition = carry[1]

        return jnp.all(jnp.logical_not(transition.done))

    def body_fn(carry):
        sim_state = carry[0]
        previous_reward = carry[2]

        next_sim_state, next_transition = policy_step(env, sim_state, policy)
        sum_reward = previous_reward + next_transition.reward

        return next_sim_state, next_transition, sum_reward

    final_sim_state, _, final_reward = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    episode_length = final_sim_state.timestep
    metrics = env.metrics(final_sim_state)

    return metrics, final_reward, episode_length

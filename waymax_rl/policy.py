import jax
from jax.random import PRNGKey

from waymax_rl.utils import Transition


def policy_step(env, simulator_state, policy, key: PRNGKey):
    obs = env.observe(simulator_state)

    actions = policy(obs, key)
    episode_slice = env.step(simulator_state, actions)

    state = episode_slice.state
    transition = Transition(
        observation=obs,
        action=actions,
        reward=episode_slice.reward,
        discount=episode_slice.discount,
        next_observation=episode_slice.observation,
    )
    metrics = episode_slice.metrics

    return state, transition, metrics


def random_step(env, simulator_state, action_shape, key: PRNGKey):
    obs = env.observe(simulator_state)

    # NOTE: Hard-coded action space
    actions = jax.random.uniform(key=key, shape=action_shape, minval=-1.0, maxval=1.0)
    episode_slice = env.step(simulator_state, actions)

    state = episode_slice.state
    transition = Transition(
        observation=obs,
        action=actions,
        reward=episode_slice.reward,
        discount=episode_slice.discount,
        next_observation=episode_slice.observation,
    )

    return state, transition

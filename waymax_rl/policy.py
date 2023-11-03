import jax
from jax.random import PRNGKey

from waymax_rl.utils import Transition


def policy_step(env, env_state, policy, key: PRNGKey):
    actions = policy(env_state.observation, key)
    nstate = env.step(env_state, actions)

    return nstate, Transition(
        observation=env_state.observation,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.observation,
    )


def random_step(env, env_state, action_shape, key: PRNGKey):
    actions = jax.random.uniform(key=key, shape=action_shape, minval=-1.0, maxval=1.0)
    nstate = env.step(env_state, actions)

    return nstate, Transition(
        observation=env_state.observation,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.observation,
    )

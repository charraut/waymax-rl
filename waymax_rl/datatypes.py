from typing import NamedTuple

import flax
import jax
import optax

from waymax_rl.utils import Params


class Transition(NamedTuple):
    """Container for a transition."""

    observation: jax.Array
    action: jax.Array
    reward: jax.Array
    flag: jax.Array
    next_observation: jax.Array
    done: jax.Array


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    actor_params: Params
    critic_params: Params
    target_critic_params: Params
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    gradient_steps: jax.Array
    env_steps: jax.Array

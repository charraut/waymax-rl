import jax.numpy as jnp
from waymax.datatypes import Action, SimulatorState
from waymax.metrics import LogDivergenceMetric


def reward_follow_ego(state: SimulatorState, action: Action):
    """Follows the agent in the simulator state."""
    log_divergence_metric = LogDivergenceMetric().compute(state).value # (batch_dims, num_objects)

    reward = jnp.mean(log_divergence_metric, axis=-1) # (batch_dims,)
    reward = jnp.log(reward) / jnp.log(0.5) # (batch_dims,)
    reward = jnp.clip(reward, -5, 5) # (batch_dims,)

    return reward

import jax.numpy as jnp
from waymax.datatypes import Action, SimulatorState
from waymax.metrics import LogDivergenceMetric

def reward_follow_ego(state: SimulatorState, action: Action):
    """Follows the agent in the simulator state."""

    log_divergence_metric = LogDivergenceMetric().compute(state)

    return jnp.mean(log_divergence_metric.value, axis=-1)

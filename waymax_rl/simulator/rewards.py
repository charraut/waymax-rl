import jax
import jax.numpy as jnp
from waymax.datatypes import Action, SimulatorState
from waymax.metrics.imitation import LogDivergenceMetric


def reward_target(state: SimulatorState, action: Action) -> jnp.ndarray:
    """Reward function for the target task."""

    # Compute the log divergence metric
    log_divergence = LogDivergenceMetric.compute_log_divergence(
        state.current_sim_trajectory.xy, state.current_log_trajectory.xy,
    )
    log_divergence = jnp.sum(log_divergence, axis=(1, 2))

    # Reward is 1 if the log divergence is within a certain range, 0 otherwise
    reward = jnp.where(log_divergence <= 0.2, 1.0, 0.0)

    return reward

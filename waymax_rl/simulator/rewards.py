import jax
import jax.numpy as jnp
from waymax.datatypes import Action, SimulatorState
from waymax.metrics.imitation import LogDivergenceMetric
from waymax.metrics.overlap import OverlapMetric
from waymax.metrics.roadgraph import OffroadMetric

def reward_overlap_offroad(state: SimulatorState, action: Action) -> jnp.ndarray:
    """Reward function for the vectorize task."""

    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    scenario_overlap = OverlapMetric().compute(state).value
    scenario_offroad = OffroadMetric().compute(state).value

    # Gather the overlap and offroad values for the SDC
    sdc_overlap = jnp.take_along_axis(scenario_overlap, sdc_idx, axis=-1)[..., 0]
    sdc_offroad = jnp.take_along_axis(scenario_offroad, sdc_idx, axis=-1)[..., 0]

    factor = -1.0
    reward = sdc_overlap * factor + sdc_offroad * factor

    return reward

def reward_target(state: SimulatorState, action: Action) -> jnp.ndarray:
    """Reward function for the target task."""

    # Compute the log divergence metric
    log_divergence = LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    # Reward is 1 if the log divergence is within a certain range, 0 otherwise
    reward = jnp.where(log_divergence <= 0.2, 1.0, 0.0)

    return reward

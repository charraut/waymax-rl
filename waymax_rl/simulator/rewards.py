from waymax.datatypes import Action, SimulatorState
import jax.numpy as jnp

def reward_follow_ego(state: SimulatorState, action: Action):
    """Follows the agent in the simulator state."""
    sim_trajectory = state.current_sim_trajectory.xy
    log_trajectory = state.current_log_trajectory.xy

    distances = jnp.linalg.norm(sim_trajectory - log_trajectory, axis=-1)
    return -jnp.mean(distances, axis=-1)

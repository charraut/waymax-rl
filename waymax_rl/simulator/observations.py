import jax
import jax.numpy as jnp
from waymax.datatypes import SimulatorState
from waymax.datatypes.observation import sdc_observation_from_state


def normalize_by_meters(x: jax.Array, meters: int) -> jax.Array:
    return x / meters


def obs_global(state: SimulatorState, trajectory_length: int = 10, normalize: bool = True) -> jax.Array:
    batch_dims = state.batch_dims
    observation = sdc_observation_from_state(
        state,
        obs_num_steps=trajectory_length,
        roadgraph_top_k=500,
    )  # (num_envs, 1, num_objects, num_steps, 2)

    # Extract all the data from the observation
    trajectory = observation.trajectory.xy  # (num_envs, num_objects, num_steps, 2)
    roadgraph_static_points = observation.roadgraph_static_points.xy  # (num_envs, roadgraph_top_k, 2)
    sdc_pos = observation.pose2d.original_xy  # (num_envs, 2)
    sdc_yaw = observation.pose2d.original_yaw  # (num_envs, 1)

    # Normalize the trajectory
    if normalize:
        sdc_pos = normalize_by_meters(sdc_pos, meters=100)
        trajectory = normalize_by_meters(x=trajectory, meters=100)
        roadgraph_static_points = normalize_by_meters(roadgraph_static_points, meters=100)

    # Reshape
    trajectory = jnp.reshape(trajectory, (*batch_dims, -1))  # (num_envs, num_objects * num_steps * 2)
    roadgraph_static_points = jnp.reshape(roadgraph_static_points, (*batch_dims, -1))  # (num_envs, roadgraph_top_k * 2)
    sdc_pos = jnp.reshape(sdc_pos, (*batch_dims, -1))  # (num_envs, 2)
    sdc_yaw = jnp.reshape(sdc_yaw, (*batch_dims, -1))  # (num_envs, 1)

    return jnp.concatenate([trajectory, roadgraph_static_points, sdc_pos, sdc_yaw], axis=len(batch_dims))

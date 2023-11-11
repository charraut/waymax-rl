import jax
import jax.numpy as jnp
from waymax.datatypes import SimulatorState
from waymax.datatypes.observation import sdc_observation_from_state, transform_trajectory


def normalize_data_point(data_point: jax.Array) -> jax.Array:
    mean = jnp.mean(data_point, axis=0)
    std = jnp.std(data_point, axis=0) + 1e-6
    return (data_point - mean) / std


def obs_global(state: SimulatorState, trajectory_length: int = 10, normalize: bool = True) -> jax.Array:
    num_envs = state.batch_dims
    observation = sdc_observation_from_state(
        state,
        obs_num_steps=trajectory_length,
        roadgraph_top_k=50,
    )  # (num_envs, 1, num_objects, num_steps, 2)
    observation = jax.tree_map(lambda x: x[:, 0], observation)  # (num_envs, num_objects, num_steps, 2)

    # Extract all the data from the observation
    trajectory = observation.trajectory.xy  # (num_envs, num_objects, num_steps, 2)
    roadgraph_static_points = observation.roadgraph_static_points.xy  # (num_envs, roadgraph_top_k, 2)
    sdc_pos = observation.pose2d.original_xy  # (num_envs, 2)
    sdc_yaw = observation.pose2d.original_yaw  # (num_envs, 1)

    # Normalize the trajectory
    if normalize:
        sdc_pos = normalize_data_point(sdc_pos)
        trajectory = normalize_data_point(trajectory)
        roadgraph_static_points = normalize_data_point(roadgraph_static_points)

    # Reshape
    trajectory = jnp.reshape(trajectory, (*num_envs, -1))  # (num_envs, num_objects * num_steps * 2)
    roadgraph_static_points = jnp.reshape(roadgraph_static_points, (*num_envs, -1))  # (num_envs, roadgraph_top_k * 2)
    sdc_pos = jnp.reshape(sdc_pos, (*num_envs, -1))  # (num_envs, 2)
    sdc_yaw = jnp.reshape(sdc_yaw, (*num_envs, -1))  # (num_envs, 1)

    return jnp.concatenate([trajectory, roadgraph_static_points, sdc_pos, sdc_yaw], axis=len(num_envs))


def obs_global_with_target(state: SimulatorState, num_steps: int = 10) -> jax.Array:
    batch_dims = state.batch_dims
    observation = sdc_observation_from_state(state, obs_num_steps=num_steps, roadgraph_top_k=50)

    # Extract all the data from the observation
    sdc_pos = observation.pose2d.original_xy  # (batch_dims, 1, 2)
    sdc_yaw = observation.pose2d.original_yaw  # (batch_dims, 1, 1)\
    trajectory = observation.trajectory.xy  # (batch_dims, 1, num_objects, num_steps, 2)
    roadgraph_static_points = observation.roadgraph_static_points.xy  # (batch_dims, 1, roadgraph_top_k, 2)

    # Extract the target trajectory
    ego_idx = jnp.argmax(state.object_metadata.is_sdc)
    log_trajectory = jax.tree_map(lambda x: x[:, ego_idx], state.log_trajectory)  # (batch_dims, num_steps)
    log_trajectory = jax.tree_map(lambda x: x[:, None, ...], log_trajectory)  # (batch_dims, 1, num_steps)
    log_trajectory = transform_trajectory(log_trajectory, observation.pose2d).xy  # (batch_dims, 1, num_steps, 2)

    # current_step = state.timestep
    # log_trajectory = log_trajectory[:, :, current_step:num_steps, :]  # (batch_dims, 1, 5, 2)

    # Normalize the log_trajectory
    mean = log_trajectory.mean(axis=(2, 3), keepdims=True)
    std = log_trajectory.std(axis=(2, 3), keepdims=True) + 1e-6
    log_trajectory = (log_trajectory - mean) / std

    # Normalize the trajectory
    mean = trajectory.mean(axis=(2, 3, 4), keepdims=True)
    std = trajectory.std(axis=(2, 3, 4), keepdims=True) + 1e-6
    trajectory = (trajectory - mean) / std

    # Normalize the roadgraph
    mean = roadgraph_static_points.mean(axis=(2, 3), keepdims=True)
    std = roadgraph_static_points.std(axis=(2, 3), keepdims=True) + 1e-6
    roadgraph_static_points = (roadgraph_static_points - mean) / std

    # Reshape
    log_trajectory = jnp.reshape(log_trajectory, (*batch_dims, -1))
    trajectory = jnp.reshape(trajectory, (*batch_dims, -1))
    roadgraph_static_points = jnp.reshape(roadgraph_static_points, (*batch_dims, -1))
    sdc_pos = jnp.reshape(sdc_pos, (*batch_dims, -1))
    sdc_yaw = jnp.reshape(sdc_yaw, (*batch_dims, -1))

    return jnp.concatenate(
        [log_trajectory, trajectory, roadgraph_static_points, sdc_pos, sdc_yaw],
        axis=len(batch_dims),
    )

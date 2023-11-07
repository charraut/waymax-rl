import jax.numpy as jnp
from waymax.datatypes.observation import sdc_observation_from_state


def obs_global(state, num_steps=10):
    batch_dims = state.batch_dims[-1]
    observation = sdc_observation_from_state(state, obs_num_steps=num_steps, roadgraph_top_k=50)

    sdc_pos = observation.pose2d.original_xy
    sdc_yaw = observation.pose2d.original_yaw
    trajectory = observation.trajectory.xy
    roadgraph_static_points = observation.roadgraph_static_points.xy

    # Normalize the trajectory
    mean = trajectory.mean(axis=(2, 3, 4), keepdims=True)
    std = trajectory.std(axis=(2, 3, 4), keepdims=True)
    trajectory = (trajectory - mean) / std

    # Normalize the roadgraph
    mean = roadgraph_static_points.mean(axis=(2, 3), keepdims=True)
    std = roadgraph_static_points.std(axis=(2, 3), keepdims=True)
    roadgraph_static_points = (roadgraph_static_points - mean) / std

    # Concatenate
    trajectory = jnp.reshape(trajectory, (batch_dims, -1))
    roadgraph_static_points = jnp.reshape(roadgraph_static_points, (batch_dims, -1))
    sdc_pos = jnp.reshape(sdc_pos, (batch_dims, -1))
    sdc_yaw = jnp.reshape(sdc_yaw, (batch_dims, -1))

    return jnp.concatenate([trajectory, roadgraph_static_points, sdc_pos, sdc_yaw], axis=1)

def obs_global_eval(state, num_steps=10):
    observation = sdc_observation_from_state(state, obs_num_steps=num_steps, roadgraph_top_k=50)

    sdc_pos = observation.pose2d.original_xy
    sdc_yaw = observation.pose2d.original_yaw
    trajectory = observation.trajectory.xy
    roadgraph_static_points = observation.roadgraph_static_points.xy

    # Normalize the trajectory
    mean = trajectory.mean(axis=(1, 2, 3), keepdims=True)
    std = trajectory.std(axis=(1, 2, 3), keepdims=True)
    trajectory = (trajectory - mean) / std

    # Normalize the roadgraph
    mean = roadgraph_static_points.mean(axis=(1, 2), keepdims=True)
    std = roadgraph_static_points.std(axis=(1, 2), keepdims=True)
    roadgraph_static_points = (roadgraph_static_points - mean) / std

    # Concatenate
    trajectory = jnp.reshape(trajectory, -1)
    roadgraph_static_points = jnp.reshape(roadgraph_static_points, -1)
    sdc_pos = jnp.reshape(sdc_pos, -1)
    sdc_yaw = jnp.reshape(sdc_yaw, -1)

    return jnp.concatenate([trajectory, roadgraph_static_points, sdc_pos, sdc_yaw], axis=0)
import jax
import jax.numpy as jnp
from waymax.datatypes import SimulatorState
from waymax.datatypes.observation import sdc_observation_from_state
from waymax.utils.geometry import transform_points
from waymax.visualization import plot_observation


def normalize_by_meters(data: jax.Array, meters: int = 20) -> jax.Array:
    data = jnp.clip(data, -meters, meters)
    return data / meters


def obs_vectorize(state: SimulatorState, trajectory_length: int = 1, normalize: bool = True) -> jax.Array:
    batch_dims = state.batch_dims
    observation = sdc_observation_from_state(state, obs_num_steps=trajectory_length, roadgraph_top_k=250)

    # Extract all the data from the observation
    trajectory = observation.trajectory.xy
    roadgraph_static_points = observation.roadgraph_static_points.xy

    # Normalize the trajectory
    if normalize:
        trajectory = normalize_by_meters(trajectory)
        roadgraph_static_points = normalize_by_meters(roadgraph_static_points)

    # Reshape
    trajectory = jnp.reshape(trajectory, (*batch_dims, -1))
    roadgraph_static_points = jnp.reshape(roadgraph_static_points, (*batch_dims, -1))

    return jnp.concatenate([trajectory, roadgraph_static_points], axis=len(batch_dims))


def obs_target(state: SimulatorState, trajectory_length: int = 10, normalize: bool = False) -> jax.Array:
    batch_dims = state.batch_dims
    len_dims = len(batch_dims)

    observation_sdc = sdc_observation_from_state(state)

    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    sdc_log_xy = jnp.take_along_axis(state.log_trajectory.xy, sdc_idx[..., None, None], axis=len_dims)
    sdc_pose2d = observation_sdc.pose2d.matrix

    sdc_log_xy = transform_points(pts=sdc_log_xy, pose_matrix=sdc_pose2d)

    target = jnp.full_like(sdc_log_xy, -1)
    target = jnp.take(target, jnp.arange(trajectory_length), axis=-2)

    sdc_log_xy = jax.lax.dynamic_slice_in_dim(sdc_log_xy, state.timestep, trajectory_length, axis=-2)
    target = jax.lax.dynamic_update_slice_in_dim(target, sdc_log_xy, 0, axis=-2)

    if normalize:
        target = normalize_by_meters(target)

    return jnp.reshape(target, (*batch_dims, -1))


def obs_bev(state: SimulatorState, trajectory_length: int = 1, size_obs: int = 20, px_per_meter: int = 2) -> jax.Array:
    observation = sdc_observation_from_state(state)

    viz_config = {
        "front_x": size_obs,
        "back_x": size_obs,
        "front_y": size_obs,
        "back_y": size_obs,
        "px_per_meter": px_per_meter,
        "show_agent_id": False,
    }

    bev = plot_observation(observation, obj_idx=0, viz_config=viz_config, batch_idx=1)
    bev = jnp.mean(bev, axis=-1)

    return bev

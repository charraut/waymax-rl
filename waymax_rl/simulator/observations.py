import jax
import jax.numpy as jnp
from waymax import config
from waymax.datatypes.observation import observation_from_state


def get_observation_spec(sample_obs, observation_fn):
    sample_obs = jax.tree_map(lambda x: x[0], sample_obs)
    observation = observation_fn(sample_obs)

    return observation.shape[-1]


def obs_follow_ego(state, num_steps=10, coordinate_frame=config.CoordinateFrame.OBJECT):
    batch_dims = state.batch_dims[-1]
    observation = observation_from_state(state, obs_num_steps=num_steps)

    ego_index = jnp.nonzero(observation.is_ego, size=batch_dims)[1]

    trajectory = observation.trajectory.xy
    batch_indices = jnp.arange(batch_dims)

    ego_trajectory = trajectory[batch_indices, ego_index]
    ego_trajectory = jnp.reshape(ego_trajectory, (batch_dims, -1))

    norms = jnp.linalg.norm(ego_trajectory, axis=1, keepdims=True) 
    normalized_ego_trajectory = ego_trajectory / norms

    return normalized_ego_trajectory

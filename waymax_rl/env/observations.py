import jax
from waymax.datatypes.observation import observation_from_state


def get_observation_spec(sample_obs):
    num_envs = sample_obs.shape[1]
    _sample_obs = jax.tree_map(lambda x: x[0], sample_obs)
    _sample_obs = jax.tree_map(lambda x: x.reshape((num_envs, -1)), _sample_obs)

    return _sample_obs.shape[-1]


def custom_obs(state):
    obs = observation_from_state(state)
    traj = obs.trajectory.xy

    if len(traj.shape) == 5:
        num_envs = traj.shape[0]
        traj = jax.tree_map(lambda x: x.reshape((num_envs, -1)), traj)
    else:
        batch_size = traj.shape[0]
        num_envs = traj.shape[1]
        traj = jax.tree_map(lambda x: x.reshape((batch_size, num_envs, -1)), traj)

    return traj

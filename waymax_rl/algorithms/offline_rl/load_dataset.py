import jax.numpy as jnp

def load_dataset_bc(path: str):
    data = jnp.load(path)
    obs = data['arr_0']
    acs = data['arr_1']
    return obs, acs

def load_dataset_offline_rl(path: str):
    data = jnp.load(path)
    obs = data['arr_0']
    acs = data['arr_1']
    rews = data['arr_2']
    next_obs = data['arr_4']
    done = data['arr_5']
    return obs, acs, rews, next_obs, done
import argparse
import os 
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax

from waymax.agents.expert import infer_expert_action
from waymax.config import DataFormat, DatasetConfig
from waymax.dataloader import simulator_state_generator

from waymax_rl.simulator.env import EnvState, WaymaxBaseEnv
from waymax_rl.algorithms.utils.buffers import ReplayBuffer, ReplayBufferState
from waymax_rl.datatypes import Transition
from waymax_rl.constants import WOD_1_1_0_TRAINING_BUCKET
from waymax_rl.simulator import create_bicycle_env

@flax.struct.dataclass
class DataState:
    inputs: jax.Array
    outputs: jax.Array

def parse_args():
    parser = argparse.ArgumentParser()

    # Data 
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_num_objects", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--trajectory_length", type=int, default=1)
    parser.add_argument("--path_dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()

def make_simulator_state_generator(
    path: str,
    max_num_objects: int,
    seed: int = None,
    batch_dims: tuple = (),
    distributed: bool = True,
):
    return simulator_state_generator(
        DatasetConfig(
            path=path,
            max_num_rg_points=20000,
            data_format=DataFormat.TFRECORD,
            max_num_objects=max_num_objects,
            batch_dims=batch_dims,
            distributed=distributed,
            shuffle_seed=seed,
        ),
    )

def expert_step(
    env: "WaymaxBaseEnv",
    env_state: "EnvState",
    key: jax.random.PRNGKey,
) -> tuple:

    # Generating actions within the specified bounds
    expert_action = infer_expert_action(env_state.simulator_state, env._dynamics_model)
    # ego_idx = jnp.argmax(env_state.simulator_state.object_metadata.is_sdc)
    # expert_action = jax.tree_map(lambda x:x[:,:,ego_idx], expert_action)

    env_state, transition = env.step(env_state, expert_action.data)

    return env_state, transition


def generate_dataset(environment, args, path_to_dataset):
    
    num_devices = jax.local_device_count()
    # Environment
    env = environment

    data_generator = make_simulator_state_generator(
        path=args.path_dataset,
        max_num_objects=args.max_num_objects,
        batch_dims=(args.num_episodes, args.num_envs),
        seed=args.seed,
    )

    sample_simulator_state = env.reset(next(data_generator)) # Dims = (num_envs, num_devices from distributed True auto)

    # Observation & action spaces dimensions
    obs_size = env.observation_spec(sample_simulator_state)
    action_size = env.action_spec().data.shape[0]
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")

    # Create Replay Buffer
    replay_buffer = ReplayBuffer(
        buffer_size=args.buffer_size // num_devices,
        batch_size=args.batch_size // num_devices,
        dummy_data_sample=Transition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            flag=0.0,
            next_observation=jnp.zeros((obs_size,)),
            done=0.0,
        ),
    )

    def fill_replay_buffer(
        batch_scenarios,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        # TODO: Remove while loop and use scan instead for parallelism
        def run_expert_step(carry):
            env_state, buffer_state, key = carry
            key, step_key = jax.random.split(key)

            env_state, transition = expert_step(env, env_state, step_key)
            buffer_state = replay_buffer.insert(buffer_state, transition, env_state.mask)

            return env_state, buffer_state, key

        def run_episode(carry, simulator_state):
            def cond_fn(carry):
                env_state = carry[0]
                return jnp.any(env_state.mask)

            buffer_state, key = carry
            env_state = env.reset(simulator_state)

            _, buffer_state, key = jax.lax.while_loop(cond_fn, run_expert_step, (env_state, buffer_state, key))

            return (buffer_state, key), None

        buffer_state, _ = jax.lax.scan(
            run_episode,
            (buffer_state, key),
            batch_scenarios,
            length=args.num_episodes,
        )[0]

        return buffer_state

    fill_replay_buffer = jax.pmap(fill_replay_buffer, axis_name="i")
    
    rng = jax.random.PRNGKey(0)
    rng, rb_key = jax.random.split(rng, 2)

    # Create and initialize the replay buffer
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_devices))

    while jnp.all(buffer_state.sample_position < 20000):
        rng, fill_key = jax.random.split(rng)
        fill_keys = jax.random.split(fill_key, num_devices)

        buffer_state = fill_replay_buffer(
            next(data_generator),
            buffer_state,
            fill_keys,
        )

    # Save replay buffer in NPZ file for BC 
    buffer_state = jax.tree_map(lambda x: x[0], buffer_state)
    dataset_to_save = replay_buffer.sample(buffer_state)
    jnp.savez(path_to_dataset, *dataset_to_save[1])

if __name__ == '__main__':
    _args = parse_args()
    # save_args 
    if _args.path_dataset is None:
        _args.path_dataset = WOD_1_1_0_TRAINING_BUCKET
           
    env = create_bicycle_env(
        max_num_objects=_args.max_num_objects,
        trajectory_length=_args.trajectory_length,
    )

    path_to_dataset = 'data/expert_dataset.npz'
    generate_dataset(
        environment=env,
        args=_args,
        path_to_dataset=path_to_dataset
    )

from functools import partial

import jax
import jax.numpy as jnp
from waymax.visualization import plot_simulator_state

from waymax_rl.algorithms.sac import make_sac_networks
from waymax_rl.algorithms.utils.networks import make_inference_fn
from waymax_rl.constants import WOD_1_1_0_TRAINING_BUCKET as PATH_DATASET
from waymax_rl.pipeline import make_simulator_state_generator, policy_step
from waymax_rl.simulator import create_bicycle_env
from waymax_rl.utils import get_model_path, load_args, load_params, select_run_path, write_video


def load_model(env, scenario, args, model_path):
    obs_size = env.observation_spec(env.reset(scenario))
    action_size = env.action_spec().data.shape[0]

    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        learning_rate=args.learning_rate,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    make_policy = make_inference_fn(sac_network)

    params = load_params(model_path)
    policy = make_policy(params, deterministic=True)

    return policy


def eval_policy(env, args, scenarios, model_path, run_path, render=False):
    policy = load_model(env, scenarios, args, model_path)

    jitted_policy_step = jax.jit(partial(policy_step, env=env, policy=policy))
    jitted_reset = jax.jit(env.reset)
    count = 0

    list_actions = []

    for i in range(args.num_episode_per_epoch):
        env_state = jitted_reset(jax.tree_map(lambda x: x[i], scenarios))
        episode_reward, episode_length = 0, 0
        list_images = []
        done = False
        list_images.append(plot_simulator_state(env_state.simulator_state, use_log_traj=True, batch_idx=0))

        while not done:
            env_state, transition, actions = jitted_policy_step(env_state)

            list_actions.append(actions)

            done = transition.done[0]
            episode_reward += jnp.mean(transition.reward)
            episode_length += 1
            list_images.append(plot_simulator_state(env_state.simulator_state, use_log_traj=True, batch_idx=0))


        print(f"Episode {i + 1} / reward: {episode_reward} - length: {episode_length}")

        if episode_length < 80:
            print(f"metrics: {env_state.metrics}")
            count += 1

        print()
        if render:
            write_video(run_path, list_images, i + 1)

    # Save actions
    list_actions = jnp.stack(list_actions, axis=0)
    jnp.save(f"actions.npy", list_actions)

    print(f"Percentage of episode with length < 80: {count / args.num_episode_per_epoch}")


if __name__ == "__main__":
    run_path = select_run_path("runs/")
    model_path = get_model_path(run_path)
    args = load_args(run_path)

    args.max_num_objects = 128

    env = create_bicycle_env(
        max_num_objects=args.max_num_objects,
        trajectory_length=args.trajectory_length,
    )

    args.num_episode_per_epoch = 1

    scenarios = next(
        make_simulator_state_generator(
            path=PATH_DATASET,
            max_num_objects=args.max_num_objects,
            batch_dims=(args.num_episode_per_epoch, 1),
            seed=0,
            distributed=False,
        ),
    )

    eval_policy(env, args, scenarios, model_path, run_path, render=True)

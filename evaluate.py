from functools import partial

import jax
import jax.numpy as jnp
from waymax.visualization import plot_simulator_state

from waymax_rl.algorithms.online_rl.sac import make_sac_networks
from waymax_rl.algorithms.offline_rl.networks import make_bc_networks
from waymax_rl.algorithms.utils.networks import make_inference_fn
from waymax_rl.constants import WOD_1_1_0_VALIDATION_BUCKET
from waymax_rl.pipeline import make_simulator_state_generator, policy_step
from waymax_rl.simulator import create_bicycle_env
from waymax_rl.utils import get_model_path, load_args, load_params, select_run_path, write_video


def load_model(env, scenario, model_name, args, model_path):
    obs_size = env.observation_spec(env.reset(scenario))
    action_size = env.action_spec().data.shape[0]

    if model_name == "SAC":
        sac_network = make_sac_networks(
            observation_size=obs_size,
            action_size=action_size,
            learning_rate=args.learning_rate,
            actor_layers=args.actor_layers,
            critic_layers=args.critic_layers,
        )
        make_policy = make_inference_fn(sac_network)
    elif model_name == "BC":
        bc_network = make_bc_networks(
            observation_size=obs_size,
            action_size=action_size,
            learning_rate=args.learning_rate,
            hidden_layers=args.hidden_layers,
        )
        make_policy = make_inference_fn(bc_network)
        
    params = load_params(model_path)
    policy = make_policy(params, deterministic=True)

    return policy


def eval_policy(env, args, model_name, scenarios, model_path, run_path, render=False):
    policy = load_model(env, scenarios, model_name, args, model_path)

    jitted_policy_step = jax.jit(partial(policy_step, env=env, policy=policy))
    jitted_reset = jax.jit(env.reset)

    max_episodes = args.num_episode_per_epoch if model_name == "SAC" else args.num_epochs
    for i in range(max_episodes):
        env_state = jitted_reset(jax.tree_map(lambda x: x[i], scenarios))
        episode_reward, episode_length = 0, 0
        list_images = []
        done = False

        while not done:
            list_images.append(plot_simulator_state(env_state.simulator_state, use_log_traj=False, batch_idx=0))
            env_state, transition = jitted_policy_step(env_state)

            done = transition.done
            episode_reward += jnp.mean(transition.reward)
            episode_length += 1

        print(f"Episode {i + 1} / reward: {episode_reward} - length: {episode_length}")
        if render:
            write_video(run_path, list_images, i + 1)


if __name__ == "__main__":
    run_path = select_run_path("runs/")
    model_path = get_model_path(run_path)
    args = load_args(run_path)

    args.path_dataset = WOD_1_1_0_VALIDATION_BUCKET

    env = create_bicycle_env(
        max_num_objects=args.max_num_objects,
        trajectory_length=args.trajectory_length,
    )

    num_episodes_eval = 20

    scenarios = next(
        make_simulator_state_generator(
            path=args.path_dataset,
            max_num_objects=args.max_num_objects,
            batch_dims=(num_episodes_eval, 1),
            seed=args.seed,
            distributed=False,
        ),
    )

    eval_policy(env, args, "BC", scenarios, model_path, run_path, render=True)

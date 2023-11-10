import os
from functools import partial

import jax
import mediapy
from jax.random import PRNGKey, split
from waymax import visualization

from waymax_rl.algorithms.sac import make_sac_networks
from waymax_rl.algorithms.utils.networks import make_inference_fn
from waymax_rl.simulator.env import WaymaxBicycleEnv
from waymax_rl.simulator.observations import obs_global_eval as obs_global
from waymax_rl.utils import load_args, load_params


def load_model(env, args, path_to_model):
    obs_size = 263  # env.observation_spec(eval_mode=True)  TO DO
    action_size = env.action_spec().data.shape[0]
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")

    # Builds the SAC networks
    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    # Builds the FWD function of the SAC Policy
    make_policy = make_inference_fn(sac_network)

    params = load_params(path_to_model)
    params = jax.tree_map(lambda x: x[0], params)

    return make_policy(params, deterministic=True), action_size


def eval_policy(env, args, path_to_model, nb_episodes=5, render=False):
    rng = PRNGKey(args.seed)
    rng, key = split(rng)

    # Load model
    if path_to_model:
        policy, action_size = load_model(env, args, path_to_model)
        infer_model = True
    else:
        policy = None
        infer_model = False

    for i in range(nb_episodes):
        # Evaluate policy on this scenario
        eval_policy_one_episode(env, policy, key, action_size, i + 1, infer_model, render)


def eval_policy_one_episode(env, policy, key, action_size, idx, infer_model, render):
    key, new_key = split(key)

    # Scenario_state: init state of one scenario
    simulator_state = env.init(env.iter_scenario)
    imgs_ep = []
    ep_cumul_rew = 0
    done = False

    step = 0

    while not done:
        if infer_model:
            obs = env.observe(simulator_state)
            actions = policy(obs, key)
        else:
            # random policy
            actions = jax.random.uniform(key=new_key, shape=(action_size,), minval=-1.0, maxval=1.0)

        episode_slice = env.step(simulator_state, actions)
        simulator_state = episode_slice.state

        done = episode_slice.done
        ep_cumul_rew += episode_slice.reward
        key, new_key = split(key)

        if render:
            imgs_ep.append(visualization.plot_simulator_state(simulator_state, use_log_traj=False))

        step += 1

    if render:
        mediapy.write_video(
            "waymax_rl/evaluate/eval_scenarios/eval_" + str(idx) + ".mp4",
            imgs_ep,
            fps=10,
        )

    print("Cumulative reward of episode " + str(idx) + " : ", ep_cumul_rew)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Load args from the training
    path_to_trained_model = "models/"
    _args = load_args(path_to_trained_model)

    env = WaymaxBicycleEnv(
        max_num_objects=_args.max_num_objects,
        observation_fn=partial(obs_global, num_steps=_args.trajectory_length),
        eval_mode=True,
    )

    path_to_model = path_to_trained_model + "model_9980.pkl"
    eval_policy(env, _args, path_to_model, nb_episodes=10, render=True)

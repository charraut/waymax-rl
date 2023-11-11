import os
from random import randint

import jax
import mediapy
from jax.random import PRNGKey, split
from waymax.visualization import plot_simulator_state

from waymax_rl.algorithms.sac import make_sac_networks
from waymax_rl.algorithms.utils.networks import make_inference_fn
from waymax_rl.policy import policy_step, random_step
from waymax_rl.simulator import create_bicycle_env
from waymax_rl.utils import load_args, load_params

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def load_model(env, args, path_to_model):
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    make_policy = make_inference_fn(sac_network)

    params = load_params(path_to_model)
    params = jax.tree_map(lambda x: x[0], params)

    action_shape = (action_size,)
    policy = make_policy(params, deterministic=True)

    return policy, action_shape


def random_state(env):
    for _ in range(randint(0, 10)):
        sim_state = env.reset()

    return sim_state


def eval_policy(env, args, path_to_model, run_path, nb_episodes=5, render=False):
    rng = PRNGKey(args.seed)
    rng, key = split(rng)

    sim_state = env.init(seed=randint(0, 1000))

    if path_to_model:
        policy, action_shape = load_model(env, args, path_to_model)
        infer_model = True
    else:
        policy = None
        infer_model = False

    for i in range(nb_episodes):
        sim_state = random_state(env)
        episode_reward, episode_images = run_episode(env, sim_state, policy, key, action_shape, infer_model)

        print("Cumulative reward of episode " + str(i) + " : ", episode_reward)
        if render:
            write_video(run_path, episode_images, i + 1)


def run_episode(env, sim_state, policy, key, action_shape, infer_model):
    episode_images = []
    episode_reward = 0
    done = False

    _sim_state = sim_state

    while not done:
        episode_images.append(plot_simulator_state(_sim_state, use_log_traj=False, batch_idx=1))

        if infer_model:
            _sim_state, transition, _ = policy_step(env, _sim_state, policy)
        else:
            key, subkey = split(key)
            _sim_state, transition = random_step(env, _sim_state, action_shape, subkey)

        done = transition.done
        episode_reward += transition.reward

    return episode_reward, episode_images


def write_video(path, episode_images, idx):
    video_path = path + "mp4/"

    if not os.path.exists(video_path):
        os.makedirs(video_path)

    mediapy.write_video(video_path + "eval_" + str(idx) + ".mp4", episode_images, fps=10)


def get_model_path(path_to_model, model_name: str = ""):
    # If no model name is provided, use the last .pkl model
    if model_name == "":
        # Filter to get only files with .pkl extension
        pkl_files = [f for f in os.listdir(path_to_model) if f.endswith(".pkl")]

        if pkl_files:
            # model_name is the last file with .pkl extension
            model_name = pkl_files[-1]
            print("Model name: ", model_name)
        else:
            print("No .pkl files found in the directory")
            model_name = None

    return path_to_model + model_name


if __name__ == "__main__":

    # Load args from the training
    run_name = "SAC_11-11_11:48:58"
    run_path = "runs/" + run_name + "/"

    model_path = get_model_path(run_path)
    args = load_args(run_path)

    env = create_bicycle_env(
        max_num_objects=args.max_num_objects,
        trajectory_length=args.trajectory_length,
    )

    eval_policy(env, args, model_path, run_path, nb_episodes=10, render=True)

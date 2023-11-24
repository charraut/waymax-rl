import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from pathlib import Path
from random import randint

import mediapy
import jax
from jax.random import PRNGKey, split
from waymax.visualization import plot_simulator_state

from waymax_rl.algorithms.sac import make_sac_networks
from waymax_rl.algorithms.utils.networks import make_inference_fn
from waymax_rl.constants import WOD_1_1_0_TRAINING_BUCKET, WOD_1_0_0_TRAINING_BUCKET
from waymax_rl.policy import policy_step, random_step
from waymax_rl.simulator import create_bicycle_env
from waymax_rl.utils import load_args, load_params
from waymax_rl.utils import make_simulator_state_generator_eval


def load_model(env, evaluation_states, args, model_path):
    obs_size = env.observation_spec(evaluation_states)
    action_size = env.action_spec().data.shape[0]

    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    make_policy = make_inference_fn(sac_network)

    params = load_params(model_path)

    action_shape = (action_size,)
    policy = make_policy(params, deterministic=True)

    return policy, action_shape


def eval_policy(env, args, scenarios, model_path, run_path, render=False):
    rng = PRNGKey(args.seed)
    rng, key = split(rng)

    evaluation_states = env.reset(scenarios).simulator_state

    if model_path:
        policy, action_shape = load_model(env, evaluation_states, args, model_path)
        infer_model = True
    else:
        policy = None
        infer_model = False

    for i in range(args.num_episode_per_epoch):
        sim_state = env.reset(jax.tree_map(lambda x: x[i], scenarios))
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
        episode_images.append(plot_simulator_state(_sim_state.simulator_state, use_log_traj=False, batch_idx=0))

        if infer_model:
            _sim_state, transition = policy_step(env, _sim_state, policy)
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


def get_model_path(model_path, model_name: str = ""):
    # If no model name is provided, use the last .pkl model
    if model_name == "":
        # Filter to get only files with .pkl extension
        pkl_files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]

        if pkl_files:
            # model_name is the last file with .pkl extension
            model_name = pkl_files[-1]
            print("Model name: ", model_name)
        else:
            print("No .pkl files found in the directory")
            return None

    return model_path + model_name


def list_folders_in_directory(directory):
    """List all folders in the given directory."""
    return [item for item in os.listdir(directory) if Path(directory, item).is_dir()]


def choose_folder(folders):
    """Prompt the user to choose a folder from the list."""
    for i, folder in enumerate(folders, start=1):
        print(f"[{i}] {folder}")

    try:
        choice = input(f"-> Enter the number of the folder you choose (default is {len(folders)}): ")
        choice = int(choice) if choice else len(folders)
        return folders[choice - 1]
    except (ValueError, IndexError):
        print("Using the last folder as default.")
        return folders[-1]


if __name__ == "__main__":
    run_path = "runs/"

    folders = list_folders_in_directory(run_path)

    if folders:
        chosen_folder = choose_folder(folders)
        chosen_folder_path = Path(run_path, chosen_folder)
        print(f"You have selected: {chosen_folder_path}")
        # Save the path in a variable
        selected_folder_path = str(chosen_folder_path) + "/"
    else:
        print("No folders found in the directory.")
        selected_folder_path = None

    model_path = get_model_path(selected_folder_path)
    args = load_args(selected_folder_path)

    if args.path_dataset is None:
        args.path_dataset = WOD_1_0_0_TRAINING_BUCKET

    env = create_bicycle_env(
        max_num_objects=args.max_num_objects,
        trajectory_length=args.trajectory_length,
    )

    data_generator = make_simulator_state_generator_eval(
        path=args.path_dataset,
        max_num_objects=args.max_num_objects,
        batch_dims=(args.num_episode_per_epoch, args.num_envs),
        seed=args.seed,
    )
    scenarios = next(data_generator)

    eval_policy(env, args, scenarios, model_path, run_path, render=True)
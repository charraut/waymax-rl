import functools
import json
import os
import pickle
from argparse import ArgumentParser
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mediapy
from etils import epath


Params = Any
Metrics = Mapping[str, jax.Array]
ActivationFn = Callable[[jax.Array], jax.Array]
Initializer = Callable[..., Any]


def print_hyperparameters(args):
    print("parameters".center(50, "="))
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("device".center(50, "="))
    print(f"jax.local_devices_to_use: {jax.local_device_count()}")
    print(f"jax.default_backend(): {jax.default_backend()}")
    print(f"jax.local_devices(): {jax.local_devices()}")


def load_params(path: str) -> Any:
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def synchronize_hosts():
    if jax.process_count() == 1:
        return

    # Make sure all processes stay up until the end of main
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, "i"), "i")(x))

    assert x[0] == jax.device_count()


def _fingerprint(x: Any) -> float:
    sums = jax.tree_util.tree_map(jnp.sum, x)

    return jax.tree_util.tree_reduce(lambda x, y: x + y, sums)


def is_replicated(x: Any, axis_name: str) -> jax.Array:
    """Returns whether x is replicated.

    Should be called inside a function pmapped along 'axis_name'
    Args:
      x: Object to check replication.
      axis_name: pmap axis_name.

    Returns:
      boolean whether x is replicated.
    """
    fp = _fingerprint(x)

    return jax.lax.pmin(fp, axis_name=axis_name) == jax.lax.pmax(fp, axis_name=axis_name)


def assert_is_replicated(x: Any, debug: Any = None):
    """Returns whether x is replicated.

    Should be called from a non-jitted code.
    Args:
      x: Object to check replication.
      debug: Debug message in case of failure.
    """
    f = functools.partial(is_replicated, axis_name="i")

    assert jax.pmap(f, axis_name="i")(x)[0], debug


def unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


# Args save & load
def save_args(args, path):
    with open(path + "/training_args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(path):
    parser = ArgumentParser()
    args = parser.parse_args()
    with open(path + "training_args.txt") as f:
        args.__dict__ = json.load(f)

    return args


def write_video(run_path, episode_images, idx):
    video_path = run_path + "mp4/"

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


def select_run_path(run_path):
    folders = list_folders_in_directory(run_path)

    if folders:
        chosen_folder = choose_folder(folders)
        chosen_folder_path = Path(run_path, chosen_folder)
        print(f"You have selected: {chosen_folder_path}")
        # Save the path in a variable
        path = str(chosen_folder_path) + "/"
    else:
        print("No folders found in the directory.")
        path = None

    return path

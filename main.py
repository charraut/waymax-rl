import argparse
import os
from collections.abc import Sequence
from datetime import datetime
from functools import partial

import jax
from tensorboardX import SummaryWriter

from waymax_rl.constants import WOD_1_1_0_TRAINING_BUCKET, WOD_1_1_0_VALIDATION_BUCKET
from waymax_rl.pipeline import run
from waymax_rl.simulator import create_bicycle_env
from waymax_rl.utils import save_args


def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--grad_updates_per_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_episode_per_epoch", type=int, default=32)
    parser.add_argument("--max_num_objects", type=int, default=16)
    parser.add_argument("--trajectory_length", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=0)
    parser.add_argument("--num_scenario_per_eval", type=int, default=16)
    # SAC
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    # Network
    parser.add_argument("--actor_layers", type=Sequence[int], default=(256, 256))
    parser.add_argument("--critic_layers", type=Sequence[int], default=(256, 256))
    # Replay Buffer
    parser.add_argument("--buffer_size", type=int, default=327_680)
    parser.add_argument("--learning_start", type=int, default=8192)
    # Misc
    parser.add_argument("--path_dataset", type=str, default=None)
    parser.add_argument("--path_dataset_eval", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_tpu", action="store_true", default=False)

    _args = parser.parse_args()

    if _args.path_dataset is None:
        _args.path_dataset = WOD_1_1_0_TRAINING_BUCKET
    if _args.path_dataset_eval is None:
        _args.path_dataset_eval = WOD_1_1_0_VALIDATION_BUCKET

    if _args.num_envs > 1:
        raise NotImplementedError("Multiple environments are not supported yet.")

    return _args


def setup_debugging(args):
    """
    Setup debugging configurations.
    """
    if args.debug_tpu:
        # Set XLA flags
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        jax.config.update("jax_platform_name", "cpu")

    args.total_timesteps = 80_000
    args.num_envs = 1
    args.grad_updates_per_step = 1
    args.batch_size = 16
    args.num_episode_per_epoch = 1
    args.save_freq = 1
    args.max_num_objects = 16
    args.trajectory_length = 2
    # SAC
    args.actor_layers = (32, 32)
    args.critic_layers = (32, 32)
    args.buffer_size = 1_000
    args.learning_start = 800

    return args


def print_metrics(num_steps, metrics, writer=None):
    """
    Print metrics and optionally write to tensorboard.
    """
    for key, value in metrics.items():
        if writer:
            writer.add_scalar(key, value, num_steps)
        print(f"{key}: {value}")
    print()


def setup_run(args):
    """
    Setup for running the experiment.
    """
    exp_name = "SAC"
    run_time = datetime.now().strftime("%d-%m_%H:%M:%S")
    path_to_save_model = f"runs/{exp_name}_{run_time}"
    writer = SummaryWriter(path_to_save_model)

    # Save hyperparameters and args
    hyperparameters_text = "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + hyperparameters_text)
    save_args(args, path_to_save_model)

    return writer, path_to_save_model


if __name__ == "__main__":
    _args = parse_args()

    if _args.debug or _args.debug_tpu:
        setup_debugging(_args)
        path_to_save_model = None
        progress = print_metrics
    else:
        writer, path_to_save_model = setup_run(_args)
        progress = partial(print_metrics, writer=writer)

    env = create_bicycle_env(
        max_num_objects=_args.max_num_objects,
        trajectory_length=_args.trajectory_length,
    )

    run(
        args=_args,
        environment=env,
        progress_fn=progress,
        checkpoint_logdir=path_to_save_model,
    )

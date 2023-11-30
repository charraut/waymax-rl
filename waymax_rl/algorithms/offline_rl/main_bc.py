import argparse
from collections.abc import Sequence
from functools import partial

from waymax_rl.algorithms.offline_rl.bc import bc_run
from waymax_rl.algorithms.offline_rl.load_dataset import load_dataset_bc
from waymax_rl.utils import setup_run, print_metrics, list_folders_in_directory

def parse_args():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--max_num_objects", type=int, default=16)
    parser.add_argument("--trajectory_length", type=int, default=1)
    parser.add_argument("--num_grad_steps_per_update", type=int, default=1)
    # BC
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_layers", type=Sequence[int], default=(256, 256))
    # Misc
    parser.add_argument("--seed", type=int, default=0)

    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    _args = parse_args()

    writer, path_to_save_model = setup_run("BC", _args)
    progress = partial(print_metrics, writer=writer)

    # BC dataset
    path = 'data/expert_dataset.npz'
    obs, acs = load_dataset_bc(path)

    # BC Training
    bc_run(obs,
           acs,
           args=_args,
           checkpoint_logdir=path_to_save_model,
        )

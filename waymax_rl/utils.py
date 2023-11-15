import functools
import json
import pickle
from argparse import ArgumentParser
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, TypeVar

import flax
import jax
import jax.numpy as jnp
import optax
from etils import epath
from waymax.config import DataFormat, DatasetConfig


Params = Any
PRNGKey = jnp.ndarray
Metrics = Mapping[str, jnp.ndarray]
NetworkType = TypeVar("NetworkType")
ReplayBufferState = Any
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]
State = TypeVar("State")
Sample = TypeVar("Sample")


class Transition(NamedTuple):
    """Container for a transition."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    flag: jnp.ndarray
    next_observation: jnp.ndarray
    done: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    actor_optimizer_state: optax.OptState
    actor_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    target_critic_params: Params
    gradient_steps: jnp.ndarray


def init_training_state(
    key: PRNGKey,
    num_devices: int,
    neural_network,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_actor, key_critic = jax.random.split(key)

    actor_params = neural_network.actor_network.init(key_actor)
    actor_optimizer_state = actor_optimizer.init(actor_params)
    critic_params = neural_network.critic_network.init(key_critic)
    critic_optimizer_state = critic_optimizer.init(critic_params)

    training_state = TrainingState(
        actor_optimizer_state=actor_optimizer_state,
        actor_params=actor_params,
        critic_optimizer_state=critic_optimizer_state,
        critic_params=critic_params,
        target_critic_params=critic_params,
        gradient_steps=jnp.zeros(()),
    )

    return jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])


def make_dataset_config(path: str, max_num_objects: int, batch_dims: tuple, seed: int):
    return DatasetConfig(
        path=path,
        max_num_rg_points=20000,
        data_format=DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
        batch_dims=batch_dims,
        distributed=True,
        shuffle_seed=seed,
    )


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
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, "batch"), "batch")(x))

    assert x[0] == jax.device_count()


def _fingerprint(x: Any) -> float:
    sums = jax.tree_util.tree_map(jnp.sum, x)

    return jax.tree_util.tree_reduce(lambda x, y: x + y, sums)


def is_replicated(x: Any, axis_name: str) -> jnp.ndarray:
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
    f = functools.partial(is_replicated, axis_name="batch")

    assert jax.pmap(f, axis_name="batch")(x)[0], debug


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

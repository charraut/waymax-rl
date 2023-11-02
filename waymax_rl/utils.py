import functools
import pickle
from collections.abc import Mapping
from typing import Any, NamedTuple, TypeVar

import flax
import jax
import jax.numpy as jnp
import optax
from etils import epath


Params = Any
PRNGKey = jnp.ndarray
Metrics = Mapping[str, jnp.ndarray]
PolicyParams = Params
NetworkType = TypeVar("NetworkType")
ReplayBufferState = Any
PMAP_AXIS_NAME = "i"


class Transition(NamedTuple):
    """Container for a transition."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray


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
    f = functools.partial(is_replicated, axis_name="i")

    assert jax.pmap(f, axis_name="i")(x)[0], debug


def unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def handle_devices(max_devices_per_host):
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()

    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)

    device_count = local_devices_to_use * jax.process_count()

    print("device".center(50, "="))
    print(f"process_id: {process_id}")
    print(f"local_devices_to_use: {local_devices_to_use}")
    print(f"device_count: {device_count}")
    print(f"jax.process_count(): {jax.process_count()}")
    print(f"jax.default_backend(): {jax.default_backend()}")
    print(f"jax.local_devices(): {jax.local_devices()}")

    return process_id, local_devices_to_use, device_count

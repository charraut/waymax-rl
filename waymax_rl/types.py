from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import jax.numpy as jnp


Params = Any
PRNGKey = jnp.ndarray
Metrics = Mapping[str, jnp.ndarray]
NetworkType = TypeVar("NetworkType")
ReplayBufferState = Any
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]
State = TypeVar("State")
Sample = TypeVar("Sample")

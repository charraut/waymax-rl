# Code source: https://github.com/google/brax/blob/main/brax/training/networks.py

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import optax
from flax import linen

from waymax_rl.types import ActivationFn, Initializer, Params, PRNGKey


class Policy(Protocol):
    def __call__(
        self,
        observation: jax.Array,
        key: PRNGKey,
    ) -> jax.Array:
        pass


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias)(
                hidden,
            )
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden


def make_actor_network(
    param_size: int,
    obs_size: int,
    actor_layers: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a policy neural_network."""
    actor_module = MLP(
        layer_sizes=list(actor_layers) + [param_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(actor_params, obs):
        return actor_module.apply(actor_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: actor_module.init(key, dummy_obs), apply=apply)


def make_critic_network(
    obs_size: int,
    action_size: int,
    critic_layers: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
) -> FeedForwardNetwork:
    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                critic = MLP(
                    layer_sizes=list(critic_layers) + [1],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(critic)
            return jnp.concatenate(res, axis=-1)

    critic_module = QModule(n_critics=n_critics)

    def apply(critic_params, obs, actions):
        return critic_module.apply(critic_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))

    return FeedForwardNetwork(init=lambda key: critic_module.init(key, dummy_obs, dummy_action), apply=apply)


def make_inference_fn(actor_critic_net):
    """Creates params and inference function."""

    def make_policy(params: Params, deterministic: bool = False) -> Policy:
        def policy(observations: jax.Array, key_sample: PRNGKey = None) -> jax.Array:
            logits = actor_critic_net.actor_network.apply(params, observations)

            if deterministic:
                return actor_critic_net.parametric_action_distribution.mode(logits)

            return actor_critic_net.parametric_action_distribution.sample(logits, key_sample)

        return policy

    return make_policy


def loss_and_pgrad(loss_fn: Callable[..., float], pmap_axis_name: str | None, has_aux: bool = False):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grad = g(*args, **kwargs)
        return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: str | None,
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
      loss_fn: The loss function.
      optimizer: The optimizer to apply gradients.
      pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
      has_aux: Whether the loss_fn has auxiliary data.

    Returns:
      A function that takes the same argument as the loss function plus the
      optimizer state. The output of this function is the loss, the new parameter,
      and the new optimizer state.
    """
    loss_and_pgrad_fn = loss_and_pgrad(loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

    def f(*args, optimizer_state):
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state

    return f

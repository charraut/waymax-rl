# Code source: https://github.com/google/brax/blob/main/brax/training/networks.py

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import flax
import jax
import jax.numpy as jnp
import optax
from flax import linen

from waymax_rl.algorithms.utils.distributions import NormalTanhDistribution, ParametricDistribution
from waymax_rl.utils import Transition


Params = Any
PolicyParams = Any
PreprocessorParams = Any
PolicyParams = tuple[PreprocessorParams, Params]
PRNGKey = jnp.ndarray
Observation = jnp.ndarray
Action = jnp.ndarray
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class Policy(Protocol):
    def __call__(
        self,
        observation: Observation,
        key: PRNGKey,
    ) -> Action:
        pass


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


@flax.struct.dataclass
class SACNetworks:
    actor_network: FeedForwardNetwork
    critic_network: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution


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
    """Creates a policy network."""
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


# Builds the SAC network (action dist, pi network, critic network)
def make_sac_networks(
    observation_size: int,
    action_size: int,
    actor_layers: Sequence[int] = (256, 256),
    critic_layers: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> SACNetworks:
    parametric_action_distribution = NormalTanhDistribution(event_size=action_size)

    actor_network = make_actor_network(
        parametric_action_distribution.param_size,
        observation_size,
        actor_layers=actor_layers,
        activation=activation,
    )

    critic_network = make_critic_network(
        observation_size,
        action_size,
        critic_layers=critic_layers,
        activation=activation,
    )

    return SACNetworks(
        actor_network=actor_network,
        critic_network=critic_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_policy(params: PolicyParams, deterministic: bool = False) -> Policy:
        def policy(observations: Observation, key_sample: PRNGKey) -> Action:
            logits = sac_networks.actor_network.apply(params, observations)

            if deterministic:
                return sac_networks.parametric_action_distribution.mode(logits)

            return sac_networks.parametric_action_distribution.sample(logits, key_sample)

        return policy

    return make_policy


def make_losses(sac_network: SACNetworks, reward_scaling: float, discount_factor: float):
    """Creates the SAC losses."""

    actor_network = sac_network.actor_network
    critic_network = sac_network.critic_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def critic_loss(
        critic_params: Params,
        actor_params: Params,
        target_critic_params: Params,
        alpha: float,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        critic_old_action = critic_network.apply(critic_params, transitions.observation, transitions.action)
        next_dist_params = actor_network.apply(actor_params, transitions.next_observation)
        next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)
        next_critic = critic_network.apply(target_critic_params, transitions.next_observation, next_action)
        next_v = jnp.min(next_critic, axis=-1) - alpha * next_log_prob
        target_critic = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * discount_factor * next_v,
        )
        critic_error = critic_old_action - jnp.expand_dims(target_critic, -1)

        # Better bootstrapping for truncated episodes
        truncation = transitions.extras["state_extras"]["truncation"]
        critic_error *= jnp.expand_dims(1 - truncation, -1)

        critic_loss = 0.5 * jnp.mean(jnp.square(critic_error))
        return critic_loss

    def actor_loss(
        actor_params: Params,
        critic_params: Params,
        alpha: float,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        dist_params = actor_network.apply(actor_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        critic_action = critic_network.apply(critic_params, transitions.observation, action)
        min_critic = jnp.min(critic_action, axis=-1)
        actor_loss = alpha * log_prob - min_critic

        return jnp.mean(actor_loss)

    return critic_loss, actor_loss


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

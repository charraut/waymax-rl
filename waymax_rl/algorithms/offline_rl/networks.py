from typing import Protocol, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from waymax_rl.algorithms.utils.distributions import NormalTanhDistribution, ParametricDistribution
from waymax_rl.algorithms.utils.networks import (
    FeedForwardNetwork,
    make_actor_network
)
from waymax_rl.utils import Params, ActivationFn, Metrics
from waymax_rl.algorithms.offline_rl.dataset_bc import Batch
from waymax_rl.algorithms.utils.losses import gradient_update_fn
from waymax_rl.datatypes import BCTrainingState

class Policy(Protocol):
    def __call__(
        self,
        observation: jax.Array,
        key: jax.random.PRNGKey,
    ) -> jax.Array:
        pass

@flax.struct.dataclass
class BCNetworks:
    actor_network: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution
    actor_optimizer: optax.GradientTransformation

def make_bc_networks(
    observation_size: int,
    action_size: int,
    learning_rate: float,
    hidden_layers: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
) -> BCNetworks:
    parametric_action_distribution = NormalTanhDistribution(event_size=action_size)

    actor_network = make_actor_network(
        parametric_action_distribution.param_size,
        observation_size,
        actor_layers=hidden_layers,
        activation=activation,
    )

    actor_optimizer = optax.adam(learning_rate=learning_rate)

    return BCNetworks(
        actor_network=actor_network,
        parametric_action_distribution=parametric_action_distribution,
        actor_optimizer=actor_optimizer,
    )

def make_loss(bc_network):
    """Creates the BC loss."""

    actor_network = bc_network.actor_network
    parametric_action_distribution = bc_network.parametric_action_distribution

    def loss(
        actor_params: Params,
        batch: Batch,
        key: jax.random.PRNGKey,
    ) -> jax.Array:
        
        dist_params = actor_network.apply(actor_params, batch.observations)

        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)

        actor_loss = ((action - batch.actions)**2).mean()
        return jnp.mean(actor_loss)

    return loss

def make_sgd_step(bc_network, learning_rate: float):
    # Optimizers
    actor_optimizer = optax.adam(learning_rate=learning_rate)

    # Create losses and grad functions for SAC losses
    loss = make_loss(
        bc_network=bc_network
    )

    actor_update = gradient_update_fn(
        loss,
        actor_optimizer,
        pmap_axis_name=None
    )

    def sgd_step(
        carry: tuple[BCTrainingState, jax.random.PRNGKey],
        batch_data,
    ) -> tuple[tuple[BCTrainingState, jax.random.PRNGKey], Metrics]:
        training_state, key = carry

        key, key_actor = jax.random.split(key, 2)

        loss, actor_params, actor_optimizer_state = actor_update(
            training_state.actor_params,
            batch_data,
            key_actor,
            optimizer_state=training_state.actor_optimizer_state,
        )

        metrics = {
            "loss": loss,
        }

        new_training_state = BCTrainingState(
            actor_optimizer_state=actor_optimizer_state,
            actor_params=actor_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
        )

        return (new_training_state, key), metrics

    return sgd_step

def make_inference_fn(actor_net):
    """Creates params and inference function."""

    def make_policy(params: Params, deterministic: bool = False) -> Policy:
        def policy(observations: jax.Array, key_sample: jax.random.PRNGKey = None) -> jax.Array:
            logits = actor_net.actor_network.apply(params, observations)

            if deterministic:
                return actor_net.parametric_action_distribution.mode(logits)

            return actor_net.parametric_action_distribution.sample(logits, key_sample)

        return policy

    return make_policy


def init_bc_policy(obs_size, action_size, learning_rate, hidden_layers):
    network = make_bc_networks(
        observation_size=obs_size,
        action_size=action_size,
        learning_rate=learning_rate,
        hidden_layers=hidden_layers,
    )
    policy_fn = make_inference_fn(network)
    learn_fn = make_sgd_step(network, learning_rate)

    return network, policy_fn, learn_fn
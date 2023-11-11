from collections.abc import Callable, Sequence
from time import perf_counter

import flax
import jax
import jax.numpy as jnp
import optax
from flax import linen

from waymax_rl.algorithms.utils.buffers import ReplayBufferState, UniformSamplingQueue
from waymax_rl.algorithms.utils.distributions import NormalTanhDistribution, ParametricDistribution
from waymax_rl.algorithms.utils.networks import (
    FeedForwardNetwork,
    gradient_update_fn,
    make_actor_network,
    make_critic_network,
    make_inference_fn,
)
from waymax_rl.policy import policy_step, random_step
from waymax_rl.simulator.env import WaymaxBaseEnv
from waymax_rl.types import ActivationFn, Metrics, Params, PRNGKey
from waymax_rl.utils import (
    TrainingState,
    Transition,
    assert_is_replicated,
    init_training_state,
    save_params,
    synchronize_hosts,
    unpmap,
)


@flax.struct.dataclass
class SACNetworks:
    actor_network: FeedForwardNetwork
    critic_network: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution


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


def make_losses(sac_network, gamma: float):
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
            transitions.reward + transitions.flag * gamma * next_v,
        )
        critic_error = critic_old_action - jnp.expand_dims(target_critic, -1)

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


def train(
    environment: WaymaxBaseEnv,
    args,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: str | None = None,
):
    start_train_func = perf_counter()

    rng = jax.random.PRNGKey(args.seed)
    rng, simulator_key = jax.random.split(rng)

    # Devices handling
    # if args.num_envs > 1:
    #     raise NotImplementedError("Multiple environments are not supported yet")

    num_devices = jax.local_device_count()
    num_prefill_actor_steps = args.learning_start // args.num_envs
    num_epoch = max(args.log_freq, 1)
    num_training_steps_per_epoch = args.total_timesteps // (num_epoch + args.num_envs)
    save_freq = num_epoch // args.num_save

    # Environment
    env = environment

    simulator_state = jax.pmap(env.init)(jnp.arange(num_devices))

    # Observation & action spaces dimensions
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    action_shape = (args.num_envs, action_size)
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")

    print("device".center(50, "="))
    print(f"num_devices: {num_devices}")
    print(f"jax.local_devices_to_use: {jax.local_device_count()}")
    print(f"jax.default_backend(): {jax.default_backend()}")
    print(f"jax.local_devices(): {jax.local_devices()}")

    # Builds the SAC networks
    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    # Builds the FWD function of the SAC Policy
    make_policy = make_inference_fn(sac_network)

    # Optimizers
    actor_optimizer = optax.adam(learning_rate=args.learning_rate)
    critic_optimizer = optax.adam(learning_rate=args.learning_rate)

    # Create Replay Buffer
    replay_buffer = UniformSamplingQueue(
        buffer_size=args.buffer_size // num_devices,
        batch_size=args.batch_size * args.grad_updates_per_step // num_devices,
        dummy_data_sample=Transition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            flag=0.0,
            next_observation=jnp.zeros((obs_size,)),
        ),
    )

    # Create losses and grad functions for SAC losses
    alpha = args.alpha
    critic_loss, actor_loss = make_losses(
        sac_network=sac_network,
        gamma=args.gamma,
    )

    actor_update = gradient_update_fn(
        actor_loss,
        actor_optimizer,
        pmap_axis_name="i",
    )
    critic_update = gradient_update_fn(
        critic_loss,
        critic_optimizer,
        pmap_axis_name="i",
    )

    def sgd_step(
        carry: tuple[TrainingState, jax.random.PRNGKey],
        transitions: Transition,
    ) -> tuple[tuple[TrainingState, jax.random.PRNGKey], Metrics]:
        training_state, key = carry

        key, key_critic, key_actor = jax.random.split(key, 3)

        critic_loss, critic_params, critic_optimizer_state = critic_update(
            training_state.critic_params,
            training_state.actor_params,
            training_state.target_critic_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.critic_optimizer_state,
        )
        actor_loss, actor_params, actor_optimizer_state = actor_update(
            training_state.actor_params,
            training_state.critic_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.actor_optimizer_state,
        )

        new_target_critic_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - args.tau) + y * args.tau,
            training_state.target_critic_params,
            critic_params,
        )

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

        new_training_state = TrainingState(
            actor_optimizer_state=actor_optimizer_state,
            actor_params=actor_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            target_critic_params=new_target_critic_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
        )

        return (new_training_state, key), metrics

    def prefill_replay_buffer(
        training_state: TrainingState,
        simulator_state,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        def f(carry, unused_t):
            training_state, simulator_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)

            simulator_state, transitions = random_step(env, simulator_state, action_shape, key)
            buffer_state = replay_buffer.insert(buffer_state, transitions)

            new_training_state = training_state.replace(
                env_steps=training_state.env_steps,
            )

            return (new_training_state, simulator_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, simulator_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    def training_step(
        training_state: TrainingState,
        simulator_state,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        experience_key, training_key = jax.random.split(key)

        policy = make_policy(training_state.actor_params)
        simulator_state, transitions, _metrics = policy_step(env, simulator_state, policy, experience_key)
        buffer_state = replay_buffer.insert(buffer_state, transitions)

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.num_envs,
        )

        buffer_state, transitions = replay_buffer.sample(buffer_state)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (args.grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )

        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        metrics = {
            "rollout/buffer_current_size": replay_buffer.size(buffer_state),
            "metrics/reward": jnp.mean(transitions.reward),
            **{f"metrics/{name}": value for name, value in _metrics.items()},
        }

        return training_state, simulator_state, buffer_state, metrics

    def training_epoch(
        training_state: TrainingState,
        simulator_state,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        def f(carry, unused_t):
            ts, ss, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, ss, bs, metrics = training_step(ts, ss, bs, k)

            return (ts, ss, bs, new_key), metrics

        (training_state, simulator_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, simulator_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        return training_state, simulator_state, buffer_state, metrics

    global_key, local_key = jax.random.split(rng)

    training_state = init_training_state(
        key=global_key,
        num_devices=num_devices,
        neural_network=sac_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    del global_key

    local_key, rb_key = jax.random.split(local_key, 2)

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name="i")
    training_epoch = jax.pmap(training_epoch, axis_name="i")
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_devices))

    # Create and initialize the replay buffer
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, num_devices)

    print("shape check".center(50, "="))
    print("buffer", buffer_state.data.shape)
    print("simulator", simulator_state.shape)
    print("prefill_keys", prefill_keys.shape)

    training_state, simulator_state, buffer_state, _ = prefill_replay_buffer(
        training_state,
        simulator_state,
        buffer_state,
        prefill_keys,
    )

    # Main training loop
    current_step = 0
    print(f"-> Pre-training: {perf_counter() - start_train_func:.2f}s")
    print("training".center(50, "="))

    time_training = perf_counter()

    for i in range(num_epoch):
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, num_devices)

        t = perf_counter()
        (training_state, simulator_state, buffer_state, training_metrics) = training_epoch(
            training_state,
            simulator_state,
            buffer_state,
            epoch_keys,
        )
        epoch_training_time = perf_counter() - t

        current_step = int(unpmap(training_state.env_steps))

        training_metrics = jax.tree_util.tree_map(jnp.mean, training_metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)

        epoch_training_time = perf_counter() - t
        sps = int((args.num_envs * num_training_steps_per_epoch) / epoch_training_time)

        training_metrics = {
            "rollout/sps": sps,
            **{f"{name}": jnp.round(value, 4) for name, value in training_metrics.items()},
        }

        # Eval and logging
        if checkpoint_logdir and not i % save_freq:
            # Save current policy
            params = training_state.actor_params
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            save_params(path, params)

        print(f"-> Step {current_step}/{args.total_timesteps} - {(current_step / args.total_timesteps) * 100:.2f}%")
        progress_fn(current_step, training_metrics)

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= args.total_timesteps

    params = training_state.actor_params

    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        save_params(path, params)

    # If there was no mistakes the training_state should still be identical on all devices
    assert_is_replicated(training_state)
    synchronize_hosts()

    return (make_policy, params)

from collections.abc import Callable, Sequence
from time import perf_counter

import flax
import jax
import jax.numpy as jnp
import optax
from flax import linen
from waymax.dataloader import simulator_state_generator

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
from waymax_rl.utils import (
    ActivationFn,
    Metrics,
    Params,
    PRNGKey,
    TrainingState,
    Transition,
    assert_is_replicated,
    init_training_state,
    make_dataset_config,
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
    eval_environment: WaymaxBaseEnv,
    args,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: str | None = None,
):
    start_train_func = perf_counter()

    rng = jax.random.PRNGKey(args.seed)

    num_devices = jax.local_device_count()

    # Environment
    env = environment

    dataset = make_dataset_config(
        path=args.path_dataset,
        max_num_objects=args.max_num_objects,
        batch_dims=(args.num_episode_per_epoch, args.num_envs),
        seed=args.seed,
    )

    data_generator = simulator_state_generator(dataset)
    sample_simulator_state = env.reset(next(data_generator)).simulator_state

    scenario_length = sample_simulator_state.remaining_timesteps
    num_epoch = args.total_timesteps // args.num_episode_per_epoch
    num_steps_per_epoch = scenario_length * args.num_episode_per_epoch
    save_freq = num_epoch // args.num_save

    print("num_prefill_actor_steps", args.num_episode_per_epoch * scenario_length)
    print("num_epoch", num_epoch)
    print("num_episode_per_epoch", args.num_episode_per_epoch)
    print("scenario_length", scenario_length)

    # Observation & action spaces dimensions
    obs_size = env.observation_spec(sample_simulator_state)
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
            done=0.0,
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
        pmap_axis_name="batch",
    )
    critic_update = gradient_update_fn(
        critic_loss,
        critic_optimizer,
        pmap_axis_name="batch",
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
        )

        return (new_training_state, key), metrics

    def run_epoch(
        batch_simulator_state,
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        def run_step(carry, _x):
            training_state, env_state, buffer_state, key = carry

            experience_key, training_key, next_key = jax.random.split(key, 3)

            # Rollout step
            policy = make_policy(training_state.actor_params)
            next_env_state, transition = policy_step(env, env_state, policy, experience_key)

            mask = next_env_state.mask
            timesteps = next_env_state.timesteps
            episode_reward = next_env_state.episode_reward

            buffer_state = replay_buffer.insert(buffer_state, transition, mask)

            # TODO: Temporary fix to avoid having to deal with the mask
            # Replace new state with old state for masked environments
            # next_simulator_state = jax.tree_util.tree_map(lambda x, y: jnp.where(mask, x, y), simulator_state, next_simulator_state)

            # Learning step
            next_buffer_state, transitions = replay_buffer.sample(buffer_state)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (args.grad_updates_per_step, -1) + x.shape[1:]),
                transitions,
            )
            (next_training_state, _), sgd_metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

            metrics = {**{f"training/{name}": value for name, value in sgd_metrics.items()}}

            return (next_training_state, next_env_state, next_buffer_state, next_key), metrics

        def run_episode(carry, simulator_state):
            training_state, buffer_state, key = carry

            init_env_state = env.reset(simulator_state)

            (next_training_state, final_env_state, next_buffer_state, next_key), metrics = jax.lax.scan(
                run_step,
                (training_state, init_env_state, buffer_state, key),
                None,
                length=scenario_length,
            )

            metrics = {
                "rollout/episode_reward": final_env_state.episode_reward,
                "rollout/episode_length": final_env_state.timesteps,
                **metrics,
            }

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)

            return (next_training_state, next_buffer_state, next_key), metrics

        key, local_key = jax.random.split(key)

        (training_state, buffer_state, _), metrics = jax.lax.scan(
            run_episode,
            (training_state, buffer_state, local_key),
            batch_simulator_state,
            length=args.num_episode_per_epoch,
        )

        return training_state, buffer_state, metrics

    def prefill_replay_buffer(
        batch_simulator_state,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        key, local_key = jax.random.split(key)

        def run_random_step(carry, _x):
            env_state, buffer_state, key = carry
            step_key, next_key = jax.random.split(key)

            next_env_state, transition = random_step(env, env_state, action_shape, step_key)
            mask = next_env_state.mask

            next_buffer_state = replay_buffer.insert(buffer_state, transition, mask)

            return (next_env_state, next_buffer_state, next_key), None

        def run_episode(carry, simulator_state):
            buffer_state, key = carry
            init_env_state = env.reset(simulator_state)

            _, next_buffer_state, next_key = jax.lax.scan(
                run_random_step,
                (init_env_state, buffer_state, key),
                None,
                length=scenario_length,
            )[0]

            return (next_buffer_state, next_key), None

        buffer_state, _ = jax.lax.scan(
            run_episode,
            (buffer_state, local_key),
            batch_simulator_state,
            length=args.num_episode_per_epoch,
        )[0]

        return buffer_state

    run_epoch = jax.pmap(run_epoch, axis_name="batch")
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name="batch")

    rng, training_key, rb_key = jax.random.split(rng, 3)

    training_state = init_training_state(
        key=training_key,
        num_devices=num_devices,
        neural_network=sac_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_devices))

    # Create and initialize the replay buffer
    rng, prefill_key = jax.random.split(rng)
    prefill_keys = jax.random.split(prefill_key, num_devices)

    print("shape check".center(50, "="))
    print("buffer", buffer_state.data.shape)
    print("simulator", sample_simulator_state.shape)
    print("prefill_keys", prefill_keys.shape)

    buffer_state = prefill_replay_buffer(
        next(data_generator),
        buffer_state,
        prefill_keys,
    )

    # Main training loop
    current_step = 0
    print(f"-> Pre-training: {perf_counter() - start_train_func:.2f}s")
    print("training".center(50, "="))

    time_training = perf_counter()

    for epoch in range(num_epoch):
        rng, epoch_key = jax.random.split(rng)
        epoch_keys = jax.random.split(epoch_key, num_devices)

        t = perf_counter()
        simulator_state = next(data_generator)
        epoch_data_time = perf_counter() - t

        t = perf_counter()
        training_state, buffer_state, training_metrics = run_epoch(
            simulator_state,
            training_state,
            buffer_state,
            epoch_keys,
        )
        epoch_training_time = perf_counter() - t

        t = perf_counter()
        training_metrics = jax.tree_util.tree_map(jnp.mean, training_metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        metrics = {
            "rollout/sps": int(num_steps_per_epoch / epoch_training_time),
            **{f"{name}": jnp.round(value, 4) for name, value in training_metrics.items()},
        }

        params = unpmap(training_state.actor_params)

        # Log metrics
        if checkpoint_logdir and not epoch % save_freq:
            # Save current policy
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            save_params(path, params)

        epoch_log_time = perf_counter() - t

        current_step = epoch * scenario_length * args.num_envs
        print(f"-> Step {current_step}/{args.total_timesteps} - {(current_step / args.total_timesteps) * 100:.2f}%")
        print(f"-> Data time     : {epoch_data_time:.2f}s")
        print(f"-> Training time : {epoch_training_time:.2f}s")
        print(f"-> Log time      : {epoch_log_time:.2f}s")
        progress_fn(current_step, metrics)

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= args.total_timesteps

    final_params = unpmap(training_state.actor_params)

    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        save_params(path, final_params)

    # If there was no mistakes the training_state should still be identical on all devices
    assert_is_replicated(training_state)
    synchronize_hosts()

    return (make_policy, final_params)

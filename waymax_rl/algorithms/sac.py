from collections.abc import Callable, Sequence
from time import perf_counter

import flax
import jax
import jax.numpy as jnp
import optax
from flax import linen

from waymax_rl.algorithms.utils.buffers import ReplayBuffer, ReplayBufferState
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
    make_simulator_state_generator,
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


def make_losses(sac_network, gamma: float, alpha: float):
    """Creates the SAC losses."""

    actor_network = sac_network.actor_network
    critic_network = sac_network.critic_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def critic_loss(
        critic_params: Params,
        actor_params: Params,
        target_critic_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> jax.Array:
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
        transitions: Transition,
        key: PRNGKey,
    ) -> jax.Array:
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

    num_devices = jax.local_device_count()

    # Environment
    env = environment

    data_generator = make_simulator_state_generator(
        path=args.path_dataset,
        max_num_objects=args.max_num_objects,
        batch_dims=(args.num_episode_per_epoch, args.num_envs),
        seed=args.seed,
    )

    if args.eval_freq:
        data_generator_eval = make_simulator_state_generator(
            path="gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
            max_num_objects=args.max_num_objects,
            batch_dims=(args.num_scenario_per_eval, 1),
            seed=args.seed,
        )
        eval_scenario = next(data_generator_eval)

    sample_simulator_state = env.reset(next(data_generator)).simulator_state

    # Observation & action spaces dimensions
    obs_size = env.observation_spec(sample_simulator_state)
    action_size = env.action_spec().data.shape[0]
    action_shape = (args.num_envs, action_size)

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
    replay_buffer = ReplayBuffer(
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
    critic_loss, actor_loss = make_losses(
        sac_network=sac_network,
        gamma=args.gamma,
        alpha=args.alpha,
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
            transitions,
            key_critic,
            optimizer_state=training_state.critic_optimizer_state,
        )
        actor_loss, actor_params, actor_optimizer_state = actor_update(
            training_state.actor_params,
            training_state.critic_params,
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
        batch_simulator_state,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        def run_random_step(carry):
            env_state, buffer_state, key = carry
            key, step_key = jax.random.split(key)

            env_state, transition = random_step(env, env_state, action_shape, step_key)
            buffer_state = replay_buffer.insert(buffer_state, transition, env_state.mask)

            return env_state, buffer_state, key

        def run_episode(carry, simulator_state):
            def cond_fn(carry):
                env_state = carry[0]
                return jnp.any(env_state.mask)

            buffer_state, key = carry
            env_state = env.reset(simulator_state)

            _, buffer_state, key = jax.lax.while_loop(cond_fn, run_random_step, (env_state, buffer_state, key))

            return (buffer_state, key), None

        buffer_state, _ = jax.lax.scan(
            run_episode,
            (buffer_state, key),
            batch_simulator_state,
            length=args.num_episode_per_epoch,
        )[0]

        return buffer_state

    def run_epoch(
        batch_simulator_state,
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        def run_step(carry):
            training_state, env_state, buffer_state, key = carry
            key, step_key, training_key = jax.random.split(key, 3)

            # Rollout step
            policy = make_policy(training_state.actor_params)
            env_state, transition = policy_step(env, env_state, policy, step_key)
            buffer_state = replay_buffer.insert(buffer_state, transition, env_state.mask)

            # Learning step
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (args.grad_updates_per_step, -1) + x.shape[1:]),
                transitions,
            )
            (training_state, _), sgd_metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

            return training_state, env_state, buffer_state, key

        def run_episode(carry, simulator_state):
            def cond_fn(carry):
                env_state = carry[1]
                return jnp.any(env_state.mask)

            training_state, buffer_state, key = carry
            env_state = env.reset(simulator_state)

            training_state, env_state, buffer_state, key = jax.lax.while_loop(
                cond_fn,
                run_step,
                (training_state, env_state, buffer_state, key),
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + jnp.sum(env_state.timesteps),
            )

            metrics = {
                "rollout/episode_reward": env_state.episode_reward,
                "rollout/episode_length": env_state.timesteps,
                **{f"rollout/{name}": value for name, value in env_state.metrics.items()},
            }

            # Mean by steps
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)

            return (training_state, buffer_state, key), metrics

        (training_state, buffer_state, _), metrics = jax.lax.scan(
            run_episode,
            (training_state, buffer_state, key),
            batch_simulator_state,
            length=args.num_episode_per_epoch,
        )

        return training_state, buffer_state, metrics

    def run_evaluation(batch_simulator_state, training_state: TrainingState):
        policy = make_policy(training_state.actor_params, deterministic=True)

        def run_step(env_state):
            env_state, _ = policy_step(env, env_state, policy, None)

            return env_state

        def run_episode(_carry, simulator_state):
            def cond_fn(env_state):
                return jnp.any(env_state.mask)

            env_state = jax.lax.while_loop(
                cond_fn,
                run_step,
                env.reset(simulator_state),
            )

            metrics = {
                "eval/episode_reward": env_state.episode_reward,
                "eval/episode_length": env_state.timesteps,
                **{f"eval/{name}": value for name, value in env_state.metrics.items()},
            }

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)

            return (), metrics

        _, metrics = jax.lax.scan(
            run_episode,
            (),
            batch_simulator_state,
            length=args.num_scenario_per_eval,
        )

        return metrics

    run_epoch = jax.pmap(run_epoch, axis_name="batch")
    run_evaluation = jax.pmap(run_evaluation, axis_name="batch")
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name="batch")

    rng, training_key, rb_key = jax.random.split(rng, 3)

    training_state = init_training_state(
        key=training_key,
        num_devices=num_devices,
        neural_network=sac_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    # Create and initialize the replay buffer
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_devices))

    while jnp.all(buffer_state.sample_position < args.learning_start):
        rng, prefill_key = jax.random.split(rng)
        prefill_keys = jax.random.split(prefill_key, num_devices)

        buffer_state = prefill_replay_buffer(
            next(data_generator),
            buffer_state,
            prefill_keys,
        )

    print("shape check".center(50, "="))
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")
    print(f"buffer shape: {buffer_state.data.shape}")
    print(f"batch scenarios shape: {sample_simulator_state.shape}")
    print(f"-> Pre-training: {perf_counter() - start_train_func:.2f}s")
    print("training".center(50, "="))

    time_training = perf_counter()

    # Main training loop
    current_step = 0
    count_epoch = 0

    while current_step < args.total_timesteps:
        count_epoch += 1

        rng, epoch_key = jax.random.split(rng)
        epoch_keys = jax.random.split(epoch_key, num_devices)

        t = perf_counter()
        batch_simulator_state = next(data_generator)
        epoch_data_time = perf_counter() - t

        t = perf_counter()
        training_state, buffer_state, training_metrics = run_epoch(
            batch_simulator_state,
            training_state,
            buffer_state,
            epoch_keys,
        )
        training_metrics = jax.tree_util.tree_map(jnp.mean, training_metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        epoch_training_time = perf_counter() - t

        t = perf_counter()
        new_current_step = int(unpmap(training_state.env_steps))
        num_steps_done = new_current_step - current_step
        current_step = new_current_step

        metrics = {
            "rollout/sps": int(num_steps_done / epoch_training_time),
            **{f"{name}": value for name, value in training_metrics.items()},
        }

        # Save current policy
        if checkpoint_logdir and not count_epoch % args.save_freq:
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            save_params(path, unpmap(training_state.actor_params))

        epoch_log_time = perf_counter() - t

        t = perf_counter()
        # Evaluate current policy
        if args.eval_freq and not count_epoch % args.eval_freq:
            eval_metrics = run_evaluation(eval_scenario, training_state)
            eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), eval_metrics)
            progress_fn(current_step, eval_metrics)

        epoch_eval_time = perf_counter() - t

        print(f"-> Step {current_step}/{args.total_timesteps} - {(current_step / args.total_timesteps) * 100:.2f}%")
        print(f"-> Data time     : {epoch_data_time:.2f}s")
        print(f"-> Training time : {epoch_training_time:.2f}s")
        print(f"-> Log time      : {epoch_log_time:.2f}s")
        print(f"-> Eval time     : {epoch_eval_time:.2f}s")
        progress_fn(current_step, metrics)

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= args.total_timesteps

    # Save final policy
    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        save_params(path, unpmap(training_state.actor_params))

    # If there was no mistakes the training_state should still be identical on all devices
    assert_is_replicated(training_state)
    synchronize_hosts()

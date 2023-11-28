from collections.abc import Callable
from time import perf_counter

import jax
import jax.numpy as jnp
from waymax.config import DataFormat, DatasetConfig
from waymax.dataloader import simulator_state_generator

from waymax_rl.algorithms.sac import init_soft_actor_critic
from waymax_rl.algorithms.utils.buffers import ReplayBuffer, ReplayBufferState
from waymax_rl.datatypes import TrainingState, Transition
from waymax_rl.policy import policy_step, random_step
from waymax_rl.simulator.env import WaymaxBaseEnv
from waymax_rl.utils import Metrics, assert_is_replicated, print_hyperparameters, save_params, synchronize_hosts, unpmap


def init_training_state(
    key: jax.random.PRNGKey,
    num_devices: int,
    neural_network,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_actor, key_critic = jax.random.split(key)

    actor_params = neural_network.actor_network.init(key_actor)
    actor_optimizer_state = neural_network.actor_optimizer.init(actor_params)
    critic_params = neural_network.critic_network.init(key_critic)
    critic_optimizer_state = neural_network.critic_optimizer.init(critic_params)

    training_state = TrainingState(
        actor_optimizer_state=actor_optimizer_state,
        actor_params=actor_params,
        critic_optimizer_state=critic_optimizer_state,
        critic_params=critic_params,
        target_critic_params=critic_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
    )

    return jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])


def make_simulator_state_generator(
    path: str,
    max_num_objects: int,
    seed: int = 0,
    batch_dims: tuple = (),
    distributed: bool = True,
):
    return simulator_state_generator(
        DatasetConfig(
            path=path,
            max_num_rg_points=20000,
            data_format=DataFormat.TFRECORD,
            max_num_objects=max_num_objects,
            batch_dims=batch_dims,
            distributed=distributed,
            shuffle_seed=seed,
        ),
    )


def run(
    args,
    environment: WaymaxBaseEnv,
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

    do_evaluation = args.eval_freq > 1
    do_save = args.save_freq > 1 and checkpoint_logdir is not None

    if do_evaluation:
        data_generator_eval = make_simulator_state_generator(
            path="gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
            max_num_objects=args.max_num_objects,
            batch_dims=(args.num_scenario_per_eval, 1),
            seed=args.seed,
        )
        eval_scenario = next(data_generator_eval)

    # Observation & action spaces dimensions
    obs_size = env.observation_spec(env.reset(next(data_generator)))
    action_size = env.action_spec().data.shape[0]
    action_shape = (args.num_envs, action_size)

    # SAC
    sac_network, make_policy, sgd_step = init_soft_actor_critic(args, obs_size, action_size)

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

    def prefill_replay_buffer(
        batch_scenarios,
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
            batch_scenarios,
            length=args.num_episode_per_epoch,
        )[0]

        return buffer_state

    def run_epoch(
        batch_scenarios,
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        key: jax.random.PRNGKey,
    ):
        def run_step(carry):
            training_state, env_state, buffer_state, key = carry
            key, step_key, training_key = jax.random.split(key, 3)

            # Rollout step
            policy = make_policy(training_state.actor_params)
            env_state, transition = policy_step(env_state, env, policy, step_key)
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
            batch_scenarios,
            length=args.num_episode_per_epoch,
        )

        return training_state, buffer_state, metrics

    def run_evaluation(batch_scenarios, training_state: TrainingState):
        policy = make_policy(training_state.actor_params, deterministic=True)

        def run_step(env_state):
            env_state, _ = policy_step(env_state, env, policy, None)

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
            batch_scenarios,
            length=args.num_scenario_per_eval,
        )

        return metrics

    run_epoch = jax.pmap(run_epoch, axis_name="i")
    run_evaluation = jax.pmap(run_evaluation, axis_name="i")
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name="i")

    rng, training_key, rb_key = jax.random.split(rng, 3)

    training_state = init_training_state(
        key=training_key,
        num_devices=num_devices,
        neural_network=sac_network,
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

    print_hyperparameters(args)
    print("shape check".center(50, "="))
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")
    print(f"buffer shape: {buffer_state.data.shape}")
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
        batch_scenarios = next(data_generator)
        epoch_data_time = perf_counter() - t

        t = perf_counter()
        training_state, buffer_state, training_metrics = run_epoch(
            batch_scenarios,
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
        if do_save and not count_epoch % args.save_freq:
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            save_params(path, unpmap(training_state.actor_params))

        epoch_log_time = perf_counter() - t

        t = perf_counter()
        # Evaluate current policy
        if do_evaluation and not count_epoch % args.eval_freq:
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

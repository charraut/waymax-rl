import argparse
from collections.abc import Callable, Sequence
from time import perf_counter

import jax
import jax.numpy as jnp
import optax
from jax.random import PRNGKey, split
from tensorboardX import SummaryWriter

from waymax_rl.algorithms.sac import make_losses, make_sac_networks
from waymax_rl.algorithms.utils.buffers import ReplayBufferState, UniformSamplingQueue
from waymax_rl.algorithms.utils.networks import gradient_update_fn, make_inference_fn
from waymax_rl.policy import policy_step, random_step
from waymax_rl.simulator.env import WaymaxBicycleEnv
from waymax_rl.simulator.observations import obs_follow_ego
from waymax_rl.simulator.rewards import reward_follow_ego
from waymax_rl.types import Metrics
from waymax_rl.utils import (
    PMAP_AXIS_NAME,
    TrainingState,
    Transition,
    assert_is_replicated,
    handle_devices,
    init_training_state,
    save_params,
    synchronize_hosts,
    unpmap,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--episode_length", type=int, default=1_000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--grad_updates_per_step", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--max_num_objects", type=int, default=16)
    # SAC
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    # Network
    parser.add_argument("--actor_layers", type=Sequence[int], default=(256, 256))
    parser.add_argument("--critic_layers", type=Sequence[int], default=(256, 256))
    # Replay Buffer
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--learning_start", type=int, default=10000)
    # Misc
    parser.add_argument("--deterministic_eval", type=bool, default=True)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--reward_scaling", type=int, default=1)
    parser.add_argument("--normalize_observations", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_devices_per_host", type=int, default=1)

    args = parser.parse_args()

    return args


def train(
    environment,
    args,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: str | None = None,
):
    start_train_func = perf_counter()

    # Print parameters
    print("parameters".center(50, "="))
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Devices handling
    process_id, local_devices_to_use, device_count = handle_devices(args.max_devices_per_host)

    env_steps_per_actor_step = args.action_repeat * args.num_envs
    num_prefill_actor_steps = args.learning_start // args.num_envs
    num_epoch = max(args.log_freq - 1, 1)
    num_training_steps_per_epoch = args.total_timesteps // (num_epoch * env_steps_per_actor_step)

    # Environment
    env = environment

    rng = PRNGKey(args.seed)
    rng, key = split(rng)

    # Observation & action spaces dimensions
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    action_shape = (args.num_envs, action_size)
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")

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

    # Dummy transition (s,a,r,s') to initiate the replay buffer
    dummy_transition = Transition(
        observation=jnp.zeros((obs_size,)),
        action=jnp.zeros((action_size,)),
        reward=0.0,
        discount=0.0,
        next_observation=jnp.zeros((obs_size,)),
        extras={"state_extras": {"truncation": 0.0}, "policy_extras": {}},
    )

    # Create Replay Buffer
    replay_buffer = UniformSamplingQueue(
        buffer_size=args.buffer_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=args.batch_size * args.grad_updates_per_step // device_count,
    )

    # Create losses and grad functions for SAC losses
    alpha = args.alpha
    critic_loss, actor_loss = make_losses(
        sac_network=sac_network,
        reward_scaling=args.reward_scaling,
        discount_factor=args.discount_factor,
    )

    actor_update = gradient_update_fn(
        actor_loss,
        actor_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )
    critic_update = gradient_update_fn(
        critic_loss,
        critic_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )

    def sgd_step(
        carry: tuple[TrainingState, PRNGKey],
        transitions: Transition,
    ) -> tuple[tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_critic, key_actor = split(key, 3)

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
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        def f(carry, unused_t):
            training_state, env_state, buffer_state, key = carry
            key, new_key = split(key)

            env_state, transitions = random_step(env, env_state, action_shape, new_key)
            buffer_state = replay_buffer.insert(buffer_state, transitions)

            new_training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=PMAP_AXIS_NAME)

    def training_step(
        training_state: TrainingState,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        experience_key, training_key = split(key)

        policy = make_policy(training_state.actor_params)
        env_state, transitions = policy_step(env, env_state, policy, experience_key)
        buffer_state = replay_buffer.insert(buffer_state, transitions)

        training_state = training_state.replace(
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # Change the front dimension of transitions so 'update_step' is called grad_updates_per_step times by the scan
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (args.grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        metrics = {
            "training/buffer_current_size": replay_buffer.size(buffer_state),
            "metrics/reward": jnp.mean(transitions.reward),
            **{f"metrics/{name}": value for name, value in env_state.metrics.items()},
        }

        return training_state, env_state, buffer_state, metrics

    def training_epoch(
        training_state: TrainingState,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)

            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=PMAP_AXIS_NAME)

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        t = perf_counter()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = perf_counter() - t
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        return training_state, env_state, buffer_state, metrics

    global_key, local_key = split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    training_state = init_training_state(
        key=global_key,
        local_devices_to_use=local_devices_to_use,
        neural_network=sac_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    del global_key

    local_key, rb_key = split(local_key, 2)

    buffer_state = jax.pmap(replay_buffer.init)(split(rb_key, local_devices_to_use))
    env_state = jax.pmap(env.reset)(env.new_scenario)

    # Create and initialize the replay buffer
    prefill_key, local_key = split(local_key)
    prefill_keys = split(prefill_key, local_devices_to_use)

    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state,
        env_state,
        buffer_state,
        prefill_keys,
    )

    # Main training loop
    current_step = 0
    print(f"-> Pre-training: {perf_counter() - start_train_func:.2f}s")
    print("training".center(50, "="))

    time_training = perf_counter()

    for _ in range(num_epoch):
        time_training_epoch = perf_counter()
        epoch_key, local_key = split(local_key)
        epoch_keys = split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
            training_state,
            env_state,
            buffer_state,
            epoch_keys,
        )
        current_step = int(unpmap(training_state.env_steps))
        time_epoch_done = perf_counter() - time_training_epoch

        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                # Save current policy
                params = training_state.actor_params
                path = f"{checkpoint_logdir}/model_{current_step}.pkl"
                save_params(path, params)

            progress_fn(current_step, training_metrics)

            print(f"-> Step {current_step}/{args.total_timesteps} - {(current_step / args.total_timesteps) * 100:.2f}%")
            print(f"- Time : {time_epoch_done:.2f}s - ({training_metrics['training/sps']:.2f} steps/s)")
            print()

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= args.total_timesteps

    params = training_state.actor_params

    # If there was no mistakes the training_state should still be identical on all devices
    assert_is_replicated(training_state)
    synchronize_hosts()

    return (make_policy, params)


if __name__ == "__main__":
    _args = parse_args()

    exp_name = "SAC"
    path_to_save_model = f"runs/waymax/{exp_name}"

    t = perf_counter()
    env = WaymaxBicycleEnv(max_num_objects=_args.max_num_objects, num_envs=_args.num_envs, observation_fn=obs_follow_ego, reward_fn=reward_follow_ego,)
    print(f"-> Environment creation: {perf_counter() - t:.2f}s")

    # Metrics progression of training
    writer = SummaryWriter(path_to_save_model)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(_args).items()])),
    )

    def progress(num_steps, metrics):
        for key in metrics:
            writer.add_scalar(key, metrics[key], num_steps)
            print(f"{key}: {metrics[key]}")
        print()

    train(environment=env, args=_args, progress_fn=progress)

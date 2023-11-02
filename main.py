import argparse
import dataclasses
import os
from collections.abc import Callable, Sequence
from time import perf_counter

import jax
import jax.numpy as jnp
import optax
from jax.random import PRNGKey
from tensorboardX import SummaryWriter
from waymax import config as _config
from waymax import dataloader, dynamics
from waymax import env as _env
from waymax.datatypes import Action
from waymax.datatypes.observation import observation_from_state
from waymax.env.wrappers.brax_wrapper import BraxWrapper

from waymax_rl.algorithms.utils.buffers import ReplayBufferState, UniformSamplingQueue
from waymax_rl.algorithms.utils.networks import (
    Params,
    SACNetworks,
    gradient_update_fn,
    make_inference_fn,
    make_losses,
    make_sac_networks,
)
from waymax_rl.utils.logs import RunningStatisticsState, init_state, normalize, update
from waymax_rl.utils.utils import (
    PMAP_AXIS_NAME,
    Array,
    Metrics,
    TrainingState,
    Transition,
    assert_is_replicated,
    handle_devices,
    save_params,
    synchronize_hosts,
    unpmap,
)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def actor_step(
    env,
    env_state,
    policy,
    key: PRNGKey,
):
    """Collect data."""
    actions, _ = policy(env_state.observation, key)

    action_waymax = Action(data=actions, valid=jnp.ones_like(actions[..., 0:1], dtype=jnp.bool_))
    action_waymax.validate()

    nstate = env.step(env_state, action_waymax)

    return nstate, Transition(
        observation=env_state.observation,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.observation,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--episode_length", type=int, default=1_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--grad_updates_per_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log_freq", type=int, default=10)
    # SAC
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)  # If ent coef fixed - NOT USED
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
    parser.add_argument("--saved_model_freq", type=int, default=4)  # - NOT USED
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_devices_per_host", type=int, default=1)

    args = parser.parse_args()

    return args


# TO REMOVE OTHER PLACE
def init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = init_state(Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


def train(
    environment,
    scenarios,
    args,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: str | None = None,
):
    """SAC training."""
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

    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)

    init_scenario = next(scenarios)
    init_time_step = env.reset(init_scenario)

    # Observation & action spaces dimensions
    obs_size = get_observation_spec(init_time_step.observation)
    action_size = env.action_spec().data.shape[0]
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")

    if args.normalize_observations:
        normalize_fn = normalize

    # Builds the SAC networks
    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    # Builds the FWD function of the SAC Policy
    make_policy = make_inference_fn(sac_network)

    # Optimizers
    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=args.learning_rate)
    q_optimizer = optax.adam(learning_rate=args.learning_rate)

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

    # Create losses and grad functions for SAC losses / Ent coef losses - could be fixed alpha
    alpha_loss, critic_loss, actor_loss = make_losses(
        sac_network=sac_network,
        reward_scaling=args.reward_scaling,
        discount_factor=args.discount_factor,
        action_size=action_size,
    )

    # Update gradients for all losses
    alpha_update = gradient_update_fn(
        alpha_loss,
        alpha_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )
    critic_update = gradient_update_fn(
        critic_loss,
        q_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )
    actor_update = gradient_update_fn(
        actor_loss,
        policy_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )

    # One step of stochastic gradient descend for all params (alpha, policy params, q params)
    def sgd_step(
        carry: tuple[TrainingState, PRNGKey],
        transitions: Transition,
    ) -> tuple[tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)
        critic_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - args.tau) + y * args.tau,
            training_state.target_q_params,
            q_params,
        )

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
        )
        return (new_training_state, key), metrics

    # Collect rollout equivalent
    def get_experience(
        normalizer_params: RunningStatisticsState,
        policy_params: Params,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        policy = make_policy((normalizer_params, policy_params))
        env_state, transitions = actor_step(env, env_state, policy, key)

        # Updates the running statistics with the given batch of data

        normalizer_params = update(
            normalizer_params,
            transitions.observation,
            pmap_axis_name=PMAP_AXIS_NAME,
        )

        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    # Training step --> One step collection (s,a,r,s') + one Sgd step
    def training_step(
        training_state: TrainingState,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # Change the front dimension of transitions so 'update_step' is called grad_updates_per_step times by the scan
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (args.grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]

    def training_epoch(
        training_state: TrainingState,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
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

    # Note that this is NOT a pure jittable method --> Main training epoch function (in the loop)
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        nonlocal training_walltime
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
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
    )
    del global_key

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Function compilation
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=PMAP_AXIS_NAME)
    training_epoch = jax.pmap(training_epoch, axis_name=PMAP_AXIS_NAME)
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

    env_state = jax.pmap(env.reset)(init_scenario)

    # Create and initialize the replay buffer
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)

    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state,
        env_state,
        buffer_state,
        prefill_keys,
    )

    # Main training loop
    metrics = {}
    current_step = 0
    print(f"-> Pre-training: {perf_counter() - start_train_func:.2f}s")
    print("training".center(50, "="))

    time_training = perf_counter()
    training_walltime = perf_counter()

    for _ in range(num_epoch):
        time_training_epoch = perf_counter()
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
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
                params = unpmap((training_state.normalizer_params, training_state.policy_params))
                path = f"{checkpoint_logdir}/model_{current_step}.pkl"
                save_params(path, params)

            progress_fn(current_step, training_metrics)

            print(f"-> Step {current_step}/{args.total_timesteps} - {(current_step / args.total_timesteps) * 100:.2f}%")
            print(f"- Time : {time_epoch_done:.2f}s - ({training_metrics['training/sps']:.2f} steps/s)")
            print()

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= args.total_timesteps

    params = unpmap((training_state.normalizer_params, training_state.policy_params))

    # If there was no mistakes the training_state should still be identical on all devices
    assert_is_replicated(training_state)
    synchronize_hosts()
    return (make_policy, params, metrics)


def get_observation_spec(sample_obs):
    num_envs = sample_obs.shape[1]
    _sample_obs = jax.tree_map(lambda x: x[0], sample_obs)
    _sample_obs = jax.tree_map(lambda x: x.reshape((num_envs, -1)), _sample_obs)

    return _sample_obs.shape[-1]


def custom_obs(state):
    obs = observation_from_state(state)
    traj = obs.trajectory.xy

    if len(traj.shape) == 5:
        num_envs = traj.shape[0]
        traj = jax.tree_map(lambda x: x.reshape((num_envs, -1)), traj)
    else:
        batch_size = traj.shape[0]
        num_envs = traj.shape[1]
        traj = jax.tree_map(lambda x: x.reshape((batch_size, num_envs, -1)), traj)

    return traj


if __name__ == "__main__":
    args_ = parse_args()

    exp_name = "SAC"
    path_to_save_model = f"runs/waymax/{exp_name}"

    max_num_objects = 4

    dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)
    env_config = _config.EnvironmentConfig(max_num_objects=max_num_objects)

    scenarios = dataloader.simulator_state_generator(
        dataclasses.replace(
            _config.WOD_1_1_0_TRAINING,
            max_num_objects=max_num_objects,
            batch_dims=(args_.num_envs,),
            distributed=True,
        ),
    )
    waymax_env = _env.PlanningAgentEnvironment(dynamics_model, env_config)
    brax_env = BraxWrapper(waymax_env)
    brax_env.observe = custom_obs

    metrics_filter = ["training/sps", "training/walltime", "eval/episode_reward", "eval/sps", "eval/walltime"]

    # Metrics progression of training
    writer = SummaryWriter(path_to_save_model)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args_).items()])),
    )

    def progress(num_steps, metrics):
        for key in metrics:
            writer.add_scalar(key, metrics[key], num_steps)

    train(environment=brax_env, scenarios=scenarios, args=args_, progress_fn=progress)

from waymax import dataloader, dynamics, config, env
from waymax.datatypes.observation import observation_from_state
from waymax.env.wrappers.brax_wrapper import BraxWrapper
import dataclasses

import jax
import jax.numpy as jnp


def custom_obs(state):
    obs = observation_from_state(state)
    traj = obs.trajectory.xy
    traj = jnp.reshape(traj, (traj.shape[0], -1))

    return traj


if __name__ == "__main__":
    # Initialization
    max_num_objects = 4

    dynamics_model = dynamics.InvertibleBicycleModel()
    env_config = config.EnvironmentConfig(max_num_objects=max_num_objects)

    scenarios = dataloader.simulator_state_generator(
        dataclasses.replace(config.WOD_1_1_0_TRAINING, max_num_objects=max_num_objects)
    )

    waymax_env = env.PlanningAgentEnvironment(dynamics_model, env_config)
    brax_env = BraxWrapper(waymax_env)
    brax_env.observe = custom_obs

    # Rollout
    env_state = brax_env.reset(next(scenarios))
    action_spec = brax_env.action_spec()
    total_returns = 0

    for i in range(100):
        action = jax.tree_map(
            lambda x: jnp.ones(x.shape, dtype=x.dtype),
            action_spec,
        )

        env_state = brax_env.step(env_state, action)
        total_returns += brax_env.reward(env_state.state, action)

        print(i, action, total_returns)

    print(total_returns)

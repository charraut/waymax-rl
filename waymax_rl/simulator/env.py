from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from waymax.config import EnvironmentConfig, LinearCombinationRewardConfig
from waymax.datatypes import Action, SimulatorState
from waymax.dynamics import DynamicsModel, InvertibleBicycleModel
from waymax.env.planning_agent_environment import PlanningAgentEnvironment


@chex.dataclass
class EnvState:
    simulator_state: SimulatorState
    timesteps: jax.Array
    mask: jax.Array
    episode_reward: jax.Array


@chex.dataclass(frozen=True)
class EpisodeSlice:
    """Container class for Waymax transitions.

    Attributes:
      simulator_state: The current simulation state of shape (num_envs,).
      observation: The current observation of shape (num_envs, ...).
      reward: The reward obtained in the current transition of shape (num_envs,).
      done: A boolean array denoting the end of an episode of shape (num_envs,).
      flag: An array of flag values of shape (num_envs,).
      metrics: Optional dictionary of metrics.
      info: Optional dictionary of arbitrary logging information.
    """

    reward: jax.Array
    done: jax.Array
    flag: jax.Array
    next_env_state: EnvState
    next_observation: jax.Array
    metrics: dict[str, Any] = struct.field(default_factory=dict)
    info: dict[str, Any] = struct.field(default_factory=dict)


class WaymaxBaseEnv(PlanningAgentEnvironment):
    def __init__(
        self,
        dynamics_model: DynamicsModel,
        env_config: EnvironmentConfig,
        observation_fn: callable = None,
        reward_fn: callable = None,
    ) -> None:
        """Initializes the Waymax environment."""

        super().__init__(dynamics_model, env_config)

        if observation_fn is not None:
            self.observe = observation_fn
        if reward_fn is not None:
            self.reward = reward_fn

    def observation_spec(self, simulator_state: SimulatorState):
        observation = self.observe(simulator_state)

        return observation.shape[-1]

    def reset(self, simulator_state: SimulatorState) -> EnvState:
        """Resets the environment."""
        simulator_state = super().reset(simulator_state)

        mask = jnp.ones(simulator_state.batch_dims[-1], dtype=jnp.bool_)
        timesteps = jnp.full(simulator_state.batch_dims[-1], simulator_state.timestep)
        episode_reward = jnp.zeros(simulator_state.batch_dims[-1])
        env_state = EnvState(
            simulator_state=simulator_state,
            timesteps=timesteps,
            mask=mask,
            episode_reward=episode_reward,
        )

        return env_state

    def termination(self, simulator_state: SimulatorState) -> jax.Array:
        """Returns a boolean array denoting the end of an episode."""
        metrics = super().metrics(simulator_state)

        return jnp.logical_or(metrics["offroad"].value, metrics["overlap"].value)

    def metrics(self, simulator_state: SimulatorState) -> dict[str, jax.Array]:
        """Returns a dictionary of metrics."""
        metric_dict = super().metrics(simulator_state)

        for key, metric in metric_dict.items():
            metric_dict[key] = jnp.mean(metric.value)

        return metric_dict


class WaymaxBicycleEnv(WaymaxBaseEnv):
    def __init__(
        self,
        max_num_objects: int = 64,
        observation_fn: callable = None,
        reward_fn: callable = None,
        normalize_actions: bool = True,
    ) -> None:
        """Initializes the Waymax bibycle environment."""

        dynamics_model = InvertibleBicycleModel(normalize_actions=normalize_actions)
        env_config = EnvironmentConfig(
            max_num_objects=max_num_objects,
            rewards=LinearCombinationRewardConfig(
                rewards={
                    "overlap": -10.0,
                    "offroad": -10.0,
                },
            ),
        )

        super().__init__(
            dynamics_model,
            env_config,
            observation_fn,
            reward_fn,
        )

    def step(self, env_state: EnvState, action: jax.Array) -> EpisodeSlice:
        """Take a step in the environment."""

        # Validate and wrap the action
        waymax_action = Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
        waymax_action.validate()

        current_simulator_state = env_state.simulator_state

        # Compute the next simulator_state and observations
        next_simulator_state = super().step(current_simulator_state, waymax_action)
        next_obs = self.observe(next_simulator_state)

        # Calculate the reward and check for termination and truncation conditions
        reward = self.reward(next_simulator_state, waymax_action)
        termination = self.termination(next_simulator_state)
        truncation = self.truncation(next_simulator_state)
        metrics = self.metrics(next_simulator_state)

        done = jnp.logical_or(termination, truncation)
        flag = jnp.logical_not(termination)

        mask = jnp.logical_and(env_state.mask, jnp.logical_not(done))
        timesteps = jnp.where(env_state.mask, env_state.timesteps + 1, env_state.timesteps)
        episode_reward = jnp.where(env_state.mask, env_state.episode_reward + reward, env_state.episode_reward)

        next_env_state = env_state.replace(
            simulator_state=next_simulator_state,
            timesteps=timesteps,
            mask=mask,
            episode_reward=episode_reward,
        )

        return EpisodeSlice(
            reward=reward,
            flag=flag,
            done=termination,
            next_env_state=next_env_state,
            next_observation=next_obs,
            metrics=metrics,
        )

from typing import Any
from functools import partial
import chex
import jax
import jax.numpy as jnp
from flax import struct
from waymax.config import EnvironmentConfig, LinearCombinationRewardConfig
from waymax.datatypes import Action, SimulatorState
from waymax.dynamics import DynamicsModel, InvertibleBicycleModel
from waymax.env.planning_agent_environment import PlanningAgentEnvironment


@chex.dataclass(frozen=True)
class EpisodeSlice:
    """Container class for Waymax transitions.

    Attributes:
      state: The current simulation state of shape (num_envs,).
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
    next_state: SimulatorState
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

        self._keep_mask = None

    def observation_spec(self, state: SimulatorState):
        observation = self.observe(state)

        return observation.shape[-1]

    def reset(self, state: SimulatorState) -> SimulatorState:
        """Resets the environment."""
        self._keep_mask = jnp.ones(state.batch_dims[-1], dtype=jnp.bool_)
        simulator_state = super().reset(state)
        self._timesteps = jnp.full(state.batch_dims[-1], simulator_state.timestep)

        return simulator_state
    
    def termination(self, state: SimulatorState) -> jax.Array:
        """Returns a boolean array denoting the end of an episode."""
        metrics = super().metrics(state)

        return jnp.logical_or(metrics["offroad"].value, metrics["overlap"].value)

    def metrics(self, state: SimulatorState) -> dict[str, jax.Array]:
        """Returns a dictionary of metrics."""
        metric_dict = super().metrics(state)

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
                    "overlap": -2.0,
                    "offroad": -2.0,
                },
            ),
        )

        super().__init__(
            dynamics_model,
            env_config,
            observation_fn,
            reward_fn,
        )

    def step(self, state: SimulatorState, action: jax.Array) -> EpisodeSlice:
        """Take a step in the environment."""

        # Validate and wrap the action
        waymax_action = Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
        waymax_action.validate()

        # Compute the next state and observations
        next_state = super().step(state, waymax_action)
        next_obs = self.observe(next_state)

        # Calculate the reward and check for termination and truncation conditions
        reward = self.reward(state, waymax_action)
        termination = self.termination(next_state)
        truncation = self.truncation(next_state)
        done = jnp.logical_or(termination, truncation)

        # Put mask at True if episode is done
        self._keep_mask = jnp.logical_and(self._keep_mask, jnp.logical_not(done))

        # Determine the flag factor
        flag = jnp.logical_not(termination)

        metrics = self.metrics(next_state)

        self._timesteps = jnp.where(self._keep_mask, self._timesteps, self._timesteps + 1)

        info = {
            "masked_envs": jnp.logical_not(self._keep_mask),
            "timesteps": self._timesteps,
            "truncation": truncation,
            "termination": termination,
        }

        return EpisodeSlice(
            reward=reward,
            flag=flag,
            done=done,
            next_state=next_state,
            next_observation=next_obs,
            metrics=metrics,
            info=info,
        )

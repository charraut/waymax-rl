from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from waymax.config import DataFormat, DatasetConfig, EnvironmentConfig, LinearCombinationRewardConfig
from waymax.dataloader import simulator_state_generator
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
        path_dataset: str,
        max_num_objects: int = 64,
        num_envs: int = 1,
        observation_fn: callable = None,
        reward_fn: callable = None,
    ) -> None:
        """Initializes the Waymax environment."""

        super().__init__(dynamics_model, env_config)

        self._max_num_objects = max_num_objects
        self._num_envs = num_envs
        self._data_generator = None

        self._dataset_config = DatasetConfig(
            path=path_dataset,
            max_num_rg_points=20000,
            data_format=DataFormat.TFRECORD,
            max_num_objects=self._max_num_objects,
            batch_dims=(self._num_envs,),
        )

        if observation_fn is not None:
            self.observe = observation_fn
        if reward_fn is not None:
            self.reward = reward_fn

    def observation_spec(self):
        observation = self.observe(self.iter_scenario)

        return observation.shape[-1]

    @property
    def max_num_objects(self):
        return self._max_num_objects

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def iter_scenario(self) -> SimulatorState:
        return next(self._data_generator)

    def init(self, key: jax.random.PRNGKey) -> SimulatorState:
        """Initializes the data generator."""
        self._data_generator = simulator_state_generator(self._dataset_config)
        n_draw = jax.random.randint(key, (), 0, 10)

        return self.reset(n_draw=n_draw)

    def reset(self, n_draw: int = 0) -> SimulatorState:
        """Resets the environment by randomly drawing n times a new scenario from the dataset."""
        def cond_fn(carry):
            return carry[1] < n_draw

        def body_fn(carry):
            scenario = self.iter_scenario

            return scenario, carry[1] + 1

        scenario = jax.lax.while_loop(cond_fn, body_fn, (self.iter_scenario, 0))[0]

        return super().reset(scenario)

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
        num_envs: int = 1,
        path_dataset: str | None = None,
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
            path_dataset,
            max_num_objects,
            num_envs,
            observation_fn,
            reward_fn,
        )

    def step(self, state: SimulatorState, action: jax.Array) -> EpisodeSlice:
        """Take a step in the environment."""

        # Validate and wrap the action
        _action = Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
        _action.validate()

        # Compute the next state and observations
        next_state = super().step(state, _action)
        next_obs = self.observe(next_state)

        # Calculate the reward and check for termination and truncation conditions
        reward = self.reward(state, _action) + 1.0
        termination = self.termination(next_state)
        truncation = self.truncation(next_state)
        done = jnp.logical_or(termination, truncation)

        # Determine the flag factor
        flag = jnp.logical_not(termination)

        metrics = self.metrics(next_state)

        return EpisodeSlice(
            reward=reward,
            flag=flag,
            done=done,
            next_state=next_state,
            next_observation=next_obs,
            metrics=metrics,
        )

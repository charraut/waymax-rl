import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from waymax import config, dataloader, dynamics
from waymax.datatypes import Action, SimulatorState
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from waymax_rl.types import Metrics
from waymax_rl.utils import unpmap


@chex.dataclass(frozen=True)
class EpisodeSlice:
    """Container class for Waymax transitions.

    Attributes:
      state: The current simulation state of shape (...).
      observation: The current observation of shape (..,).
      reward: The reward obtained in the current transition of shape (...,
        num_objects).
      done: A boolean array denoting the end of an episode of shape (...).
      discount: An array of discount values of shape (...).
      metrics: Optional dictionary of metrics.
      info: Optional dictionary of arbitrary logging information.
    """

    state: SimulatorState
    observation: jax.Array
    reward: jax.Array
    done: jax.Array
    discount: jax.Array
    metrics: Metrics = struct.field(default_factory=dict)
    info: dict[str, Any] = struct.field(default_factory=dict)


class WaymaxBaseEnv(PlanningAgentEnvironment):
    def __init__(
        self,
        dynamics_model: dynamics.DynamicsModel,
        env_config: config.EnvironmentConfig,
        max_num_objects: int = 64,
        num_envs: int = 1,
        observation_fn: callable = None,
        reward_fn: callable = None,
        eval_mode: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(dynamics_model, env_config)

        self._max_num_objects = max_num_objects
        self._num_envs = num_envs
        self._eval_mode = eval_mode

        if eval_mode:
            self._scenarios = dataloader.simulator_state_generator(
                dataclasses.replace(
                    config.WOD_1_0_0_VALIDATION,
                    max_num_objects=max_num_objects,
                ),
            )
        else:
            self._scenarios = dataloader.simulator_state_generator(
                dataclasses.replace(
                    config.WOD_1_1_0_TRAINING,
                    max_num_objects=max_num_objects,
                    batch_dims=(num_envs,),
                    distributed=True,
                ),
            )

        if observation_fn is not None:
            self.observe = observation_fn
        if reward_fn is not None:
            self.reward = reward_fn

        self.reset = self._reset_eval if eval_mode else self._reset_train

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
        return next(self._scenarios)

    def init(self, state: SimulatorState) -> SimulatorState:
        return super().reset(state)

    def _reset_train(self) -> SimulatorState:
        return self.init(unpmap(self.iter_scenario))

    def _reset_eval(self) -> SimulatorState:
        return self.init(self.iter_scenario)

    def termination(self, state: SimulatorState) -> jax.Array:
        metrics = super().metrics(state)
        return jnp.logical_or(metrics["offroad"].value, metrics["overlap"].value)

    def metrics(self, state: SimulatorState):
        metric_dict = super().metrics(state)
        for key, metric in metric_dict.items():
            metric_dict[key] = jnp.mean(metric.value)

        return metric_dict


class WaymaxBicycleEnv(WaymaxBaseEnv):
    def __init__(
        self,
        max_num_objects: int = 64,
        num_envs: int = 1,
        observation_fn: callable = None,
        reward_fn: callable = None,
        normalize_actions: bool = True,
        eval_mode: bool = False,
    ) -> None:
        dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=normalize_actions)
        env_config = config.EnvironmentConfig(
            max_num_objects=max_num_objects,
            rewards=config.LinearCombinationRewardConfig(
                rewards={
                    "overlap": -20.0,
                    "offroad": -20.0,
                },
            ),
        )

        super().__init__(dynamics_model, env_config, max_num_objects, num_envs, observation_fn, reward_fn, eval_mode)

    def step(self, state: SimulatorState, action: jax.Array) -> EpisodeSlice:
        # Validate and wrap the action
        _action = Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
        _action.validate()

        # Compute the next state and observations
        next_state = super().step(state, _action)
        next_obs = self.observe(next_state)

        # Calculate the reward and check for termination and truncation conditions
        reward = self.reward(next_state, None)
        termination = self.termination(next_state)
        truncation = self.truncation(next_state)
        done = jnp.logical_or(termination, truncation)

        # Determine the discount factor
        discount = jnp.logical_not(termination)

        # Collect metrics if any
        metric_dict = self.metrics(next_state)

        next_state = jax.lax.cond(
            jnp.all(done),
            lambda: self.reset(),
            lambda: next_state,
        )

        return EpisodeSlice(
            state=next_state,
            reward=reward,
            observation=next_obs,
            discount=discount,
            done=done,
            metrics=metric_dict,
        )

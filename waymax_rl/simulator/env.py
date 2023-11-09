import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from waymax import config, dataloader, dynamics
from waymax.datatypes import Action, Observation, SimulatorState
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from waymax_rl.types import Metrics


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
    observation: Observation
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
        batch_dims: tuple = (),
        observation_fn: callable = None,
        reward_fn: callable = None,
        eval_mode: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(dynamics_model, env_config)

        self._max_num_objects = max_num_objects
        self._batch_dims = batch_dims
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
                    batch_dims=batch_dims,
                ),
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
    def batch_dims(self):
        return self._batch_dims

    @property
    def iter_scenario(self) -> SimulatorState:
        return next(self._scenarios)

    def init(self, state: SimulatorState) -> SimulatorState:
        return super().reset(state)

    def reset(self) -> EpisodeSlice:
        scenario = next(self._scenarios)
        return self.init(scenario)

    def metrics(self, state: SimulatorState):
        metric_dict = super().metrics(state)
        for key, metric in metric_dict.items():
            metric_dict[key] = jnp.mean(metric.value)

        return metric_dict


class WaymaxBicycleEnv(WaymaxBaseEnv):
    def __init__(
        self,
        max_num_objects: int = 64,
        batch_dims: int = 1,
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
                    "overlap": -1.0,
                    "offroad": -1.0,
                },
            ),
        )

        super().__init__(dynamics_model, env_config, max_num_objects, batch_dims, observation_fn, reward_fn, eval_mode)

    def step(self, state: SimulatorState, action: jax.Array) -> EpisodeSlice:
        # Validate and wrap the action
        _action = Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
        _action.validate()

        # Compute the next state and observations
        next_state = super().step(state, _action)
        obs = self.observe(next_state)

        # Calculate the reward and check for termination and truncation conditions
        reward = self.reward(state, _action)
        termination = self.termination(next_state)
        truncation = self.truncation(next_state)

        # Determine the discount factor
        discount = jnp.where(termination, 0.0, 1.0)

        # Collect metrics if any
        metric_dict = self.metrics(state)

        # next_state = jax.lax.cond(
        #     jnp.all(truncation),
        #     lambda: self.reset(),
        #     lambda: next_state,
        # )

        return EpisodeSlice(
            state=next_state,
            reward=reward,
            observation=obs,
            done=termination,
            discount=discount,
            metrics=metric_dict,
        )

import dataclasses

import jax
import jax.numpy as jnp
from waymax import config, dataloader, datatypes, dynamics
from waymax.datatypes import Action
from waymax.env.planning_agent_environment import PlanningAgentEnvironment
from waymax.env.wrappers.brax_wrapper import TimeStep

from waymax_rl.simulator.observations import get_observation_spec


class WaymaxBaseEnv(PlanningAgentEnvironment):
    def __init__(
        self,
        dynamics_model: dynamics.DynamicsModel,
        env_config: config.EnvironmentConfig,
        max_num_objects: int,
        num_envs: int,
        observation_fn: callable = None,
        reward_fn: callable = None,
    ) -> None:
        super().__init__(dynamics_model, env_config)
        self._max_num_objects = max_num_objects
        self._num_envs = num_envs

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

        self._observation_spec = get_observation_spec(self.observe(self.new_scenario))

    def observation_spec(self):
        return self._observation_spec

    @property
    def max_num_objects(self):
        return self._max_num_objects

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def new_scenario(self):
        return next(self._scenarios)

    def reset(self, state: datatypes.SimulatorState) -> TimeStep:
        initial_state = super().reset(state)

        return TimeStep(
            state=initial_state,
            observation=self.observe(initial_state),
            done=self.termination(initial_state),
            reward=jnp.zeros(state.shape + self.reward_spec().shape),
            discount=jnp.ones(state.shape + self.discount_spec().shape),
            metrics=self.metrics(initial_state),
        )

    def metrics(self, state: datatypes.SimulatorState):
        metric_dict = super().metrics(state)
        for key, metric in metric_dict.items():
            metric_dict[key] = jnp.mean(metric.value)

        return metric_dict


class WaymaxBicycleEnv(WaymaxBaseEnv):
    def __init__(
        self,
        max_num_objects: int,
        num_envs: int,
        observation_fn: callable = None,
        reward_fn: callable = None,
    ) -> None:
        dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)
        env_config = config.EnvironmentConfig(max_num_objects=max_num_objects)

        super().__init__(dynamics_model, env_config, max_num_objects, num_envs, observation_fn, reward_fn)

    def step(self, timestep: TimeStep, action: jax.Array) -> TimeStep:
        _action = Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
        _action.validate()

        next_state = super().step(timestep.state, _action)
        obs = self.observe(next_state)
        reward = self.reward(timestep.state, _action)
        termination = self.termination(next_state)
        truncation = self.truncation(next_state)
        done = jnp.logical_or(termination, truncation)
        discount = jnp.logical_not(termination).astype(jnp.float32)
        metric_dict = self.metrics(timestep.state)

        return TimeStep(
            state=next_state,
            reward=reward,
            observation=obs,
            done=done,
            discount=discount,
            metrics=metric_dict,
        )

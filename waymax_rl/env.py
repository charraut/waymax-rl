import dataclasses

import jax
import jax.numpy as jnp
from waymax import config, dataloader, dynamics
from waymax.datatypes import Action, observation_from_state
from waymax.env.planning_agent_environment import PlanningAgentEnvironment


class WaymaxBicycleEnv(PlanningAgentEnvironment):
    def __init__(self, max_num_objects: int) -> None:
        dynamics_model = dynamics.InvertibleBicycleModel()
        env_config = config.EnvironmentConfig(max_num_objects=max_num_objects)
        super().__init__(dynamics_model, env_config)

        # Initialise training scenarios and initial waymax state
        self.scenarios = dataloader.simulator_state_generator(
            dataclasses.replace(config.WOD_1_1_0_TRAINING, max_num_objects=max_num_objects),
        )
        self.state = next(self.scenarios)

        self.action_spec = super().action_spec()
        self._jitted_reward = jax.jit(self.reward)
        self._jitted_step = jax.jit(self.step)

    def reset(self):
        self.state = next(self.scenarios)
        self.state = super().reset(self.state)
        observation = self.observation_from_state_rl(self.state)

        return observation

    def step(self, action):
        action = Action(
            data=jnp.array([action[0], action[1]]),
            valid=jnp.ones(self.action_spec.valid.shape, dtype=jnp.bool_),
        )

        self.state = super().step(self.state, action)
        observation = self.observation_from_state_rl(self.state)
        reward = self.compute_reward(observation)
        done = self.is_episode_finished(observation)

        return observation, reward, done

    def is_episode_finished(self, obs):
        return False

    def compute_reward(self, obs):
        return 0

    def observation_from_state_rl(self, state):
        obs = observation_from_state(state)
        traj = obs.trajectory.xy
        traj = traj.reshape(-1)

        return traj

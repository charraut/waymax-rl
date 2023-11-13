from typing import Any

import jax
import jax.numpy as jnp
from waymax.datatypes import SimulatorState

from waymax_rl.simulator.env import EpisodeSlice, WaymaxBaseEnv


class Wrapper(WaymaxBaseEnv):
    """
    Wrapper class for WaymaxBaseEnv environment.
    Forwards calls to the wrapped environment and can be extended with additional functionality.
    """

    def __init__(self, env: WaymaxBaseEnv):
        """Initialize the Wrapper."""
        self._env = env

    def __getattr__(self, name: str) -> Any:
        """Forward all other calls to the wrapped environment."""
        return getattr(self._env, name)

    def init(self, key: jax.random.PRNGKey) -> SimulatorState:
        return self._env.init(key)

    def reset(self, n_draw: int = 1) -> SimulatorState:
        return self._env.reset(n_draw)

    def step(self, state: SimulatorState, action: jax.Array) -> EpisodeSlice:
        return self._env.step(state, action)

    def termination(self, state: SimulatorState) -> jax.Array:
        return self._env.termination(state)

    def reward(self, state: SimulatorState, action: jax.Array) -> jax.Array:
        return self._env.reward(state, action)

    def observe(self, state: SimulatorState) -> jax.Array:
        return self._env.observe(state)

    def metrics(self, state: SimulatorState) -> jax.Array:
        return self._env.metrics(state)


class AutoResetWrapper(Wrapper):
    """
    AutoResetWrapper class automatically resets the environment when an episode is done.
    Extends the functionality of the Wrapper class.
    """

    def step(self, state: SimulatorState, action: jax.Array) -> EpisodeSlice:
        """Perform a step in the environment and automatically reset if the episode is done."""
        output = super().step(state, action)

        next_state = jax.lax.cond(
            jnp.all(output.done),
            lambda: self._env.reset(),
            lambda: output.next_state,
        )

        return output.replace(next_state=next_state)

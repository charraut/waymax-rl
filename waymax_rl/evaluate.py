from functools import partial

import jax
import jax.numpy as jnp
from waymax_rl.policy import rollout


class Evaluator:
    """Class to run evaluations."""

    def __init__(self, key, eval_env, eval_policy_fn, num_eval=10):
        self._eval_env = eval_env
        self._eval_policy_fn = partial(eval_policy_fn, deterministic=True)
        self._num_eval = num_eval
        self._key = key

        self._key, init_key = jax.random.split(key)
        self._eval_env.init(init_key)

        self._vmap_rollout = jax.vmap(rollout, in_axes=(0, None, None))

    def run_evaluation(self, policy_params) -> dict[str, float]:
        """Run one episode of evaluation."""
        # eval_keys = jax.random.split(key, self._num_eval)
        sim_states = []
        for _ in range(self._num_eval):
            sim_states.append(self._eval_env.reset())

        eval_policy = self._eval_policy_fn(policy_params)

        metrics, reward, timestep = self._vmap_rollout(sim_states, self._eval_env, eval_policy)

        return {
            "eval/episode_length": timestep,
            "eval/reward": reward,
            **{f"eval/{name}": value for name, value in metrics.items()},
        }

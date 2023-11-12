import jax

from waymax_rl.policy import rollout


class Evaluator:
    """Class to run evaluations."""

    def __init__(self, eval_env, eval_policy_fn, num_eval=100):
        self._eval_env = eval_env
        self._eval_policy_fn = eval_policy_fn
        self._num_eval = num_eval

        self._vmap_rollout = jax.vmap(rollout, in_axes=(0, None, None))

    def run_evaluation(self, key, policy_params) -> dict[str, float]:
        """Run one episode of evaluation."""
        eval_keys = jax.random.split(key, self._num_eval)
        policy_eval = self._eval_policy_fn(policy_params)

        metrics, timestep = self._vmap_rollout(eval_keys, self._eval_env, policy_eval)

        return {
            "eval/episode_length": timestep,
            **{f"eval/{name}": value for name, value in metrics.items()},
        }

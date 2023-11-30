from typing import Sequence
import tqdm

import jax
import jax.numpy as jnp
import numpy as np

from waymax_rl.algorithms.offline_rl.dataset_bc import Dataset
from waymax_rl.algorithms.offline_rl.networks import init_bc_policy
from waymax_rl.datatypes import BCTrainingState
from waymax_rl.utils import save_params, unpmap

def init_training_state(
    key: jax.random.PRNGKey,
    num_devices: int,
    neural_network,
) -> BCTrainingState:
    """Inits the training state and replicates it over devices."""
    key_actor, _ = jax.random.split(key)

    actor_params = neural_network.actor_network.init(key_actor)
    actor_optimizer_state = neural_network.actor_optimizer.init(actor_params)

    training_state = BCTrainingState(
        actor_optimizer_state=actor_optimizer_state,
        actor_params=actor_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
    )
    return training_state # jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])

def bc_run(observations: jnp.array,
           actions: jnp.array,
           args,
           checkpoint_logdir: str | None = None):

    num_devices = jax.local_device_count()

    rng = jax.random.PRNGKey(args.seed)
    rng, actor_key = jax.random.split(rng)

    # Dataset obs, acs for Imitation
    data = Dataset(observations,
                   actions,
                   size=observations.shape[0])

    obs_size = observations.shape[-1]
    action_size = actions.shape[-1]

    neural_network, make_policy, learning_step = init_bc_policy(obs_size,
                                                                action_size,
                                                                args.learning_rate,
                                                                args.hidden_layers
                                                                )
    
    def run_epoch(batch_data,
                  training_state,
                  key: jax.random.PRNGKey):

        key, training_key = jax.random.split(key, 2)

        # Rollout step
        # policy = make_policy(training_state.actor_params)
        # actions = policy(batch_data.observations, key)

        # Learning step
        (training_state, _), sgd_metrics = learning_step((training_state,training_key), batch_data)

        return training_state, sgd_metrics
    
    # TO DO PMAP ALL BC 
    # run_epoch = jax.pmap(run_epoch, axis_name="i")
    
    # Init training state
    rng, training_key = jax.random.split(rng, 2)

    training_state = init_training_state(
        key=training_key,
        num_devices=num_devices,
        neural_network=neural_network,
    )

    count_epoch = 0
    while count_epoch < args.num_epochs:
        count_epoch += 1
        rng, epoch_key = jax.random.split(rng)

        # Batch samples (obs, expert acs)
        batch_data = data.sample(args.batch_size)

        training_state, loss = run_epoch(
            batch_data,
            training_state,
            epoch_key
        )

        if not (count_epoch % 100):
            print(f"-> Epochs     : {count_epoch}/{args.num_epochs}")
            print(f"-> Loss     : {loss}")

    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        save_params(path, training_state.actor_params)
    


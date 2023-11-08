import os
from jax.random import PRNGKey, split
from waymax import visualization
from waymax_rl.simulator.env import WaymaxBicycleEnv
from waymax_rl.simulator.observations import obs_global_eval, obs_global_with_target_eval
from waymax_rl.utils import load_params, load_args
from functools import partial

from waymax_rl.algorithms.sac import make_sac_networks
from waymax_rl.algorithms.utils.networks import make_inference_fn

import mediapy
import jax

def load_model(env, args, path_to_model):
    obs_size = 445 # env.observation_spec(eval_mode=True)  TO DO 
    action_size = env.action_spec().data.shape[0]
    print(f"observation size: {obs_size}")
    print(f"action size: {action_size}")

    # Builds the SAC networks
    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    # Builds the FWD function of the SAC Policy
    make_policy = make_inference_fn(sac_network)

    params = load_params(path_to_model)
    params = jax.tree_map(lambda x:x[0], params)

    return make_policy(params, deterministic=True), action_size 

def eval_policy(env, args, path_to_model, nb_episodes=5, render=False):

    rng = PRNGKey(args.seed)
    rng, key = split(rng)

    # Load model
    if path_to_model:
        policy, action_size = load_model(env, args, path_to_model)
        infer_model = True
    else:
        policy = None 
        infer_model = False

    for i in range(nb_episodes):
        # Evaluate policy on this scenario
        eval_policy_one_episode(env, policy, key, action_size, i+1, infer_model, render)

def eval_policy_one_episode(env, policy, key, action_size, idx, infer_model, render):

    key, new_key = split(key)

    # Scenario_state: init state of one scenario
    state = env.reset(env.init_scenario)
    imgs_ep = []
    ep_cumul_rew = 0
    done = False

    iter = 0 
    num_timesteps = state.state.remaining_timesteps

    while not done and iter < num_timesteps:
        if infer_model:
            obs = state.observation[None, ...]
            actions = policy(obs, key)[0, ...]
        else:
            # random policy
            actions = jax.random.uniform(key=new_key, shape=(action_size,), minval=-1.0, maxval=1.0)

        state = env.step(state, actions)
        done = state.done
        ep_cumul_rew += state.reward
        key, new_key = split(key)

        if render:
            imgs_ep.append(visualization.plot_simulator_state(state.state, use_log_traj=False))

        iter += 1

    if render:
        mediapy.write_video('/home/ttournai/Workspace/DAR/waymax-rl/waymax_rl/evaluate/eval_scenarios/eval_' + str(idx) +'.mp4',imgs_ep, fps=10)

    print('Cumulative reward of episode ' + str(idx) + ' : ', ep_cumul_rew)

if __name__ == '__main__':
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
    
    # Load args from the training 
    path_to_trained_model = '/home/ttournai/Workspace/DAR/waymax-rl/models/'
    _args = load_args(path_to_trained_model)

    env = WaymaxBicycleEnv(
        max_num_objects=_args.max_num_objects,
        num_envs=_args.num_envs,
        observation_fn=partial(obs_global_with_target_eval, num_steps=_args.trajectory_length),
        eval_mode=True
    )

    path_to_model = path_to_trained_model + 'model_1020100.pkl'
    eval_policy(env, _args, path_to_model, nb_episodes=10, render=True)


import numpy as np
# from multiagent.environment import MultiAgentEnv
# import multiagent.scenarios as scenarios
import gym
#from gym_fortattack.fortattack import make_fortattack_env
from gym_fortattack.open_fortattack import make_open_fortattack_env

def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs


# def make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, num_steps):
#     env = make_fortattack_env(num_steps)
#     return env

def make_open_env(args, num_steps, team_size, active_agents, num_freeze_steps, reward_scheme="normal", team_mode=None, seed=100, agent_type=-1):
    env = make_open_fortattack_env(
        args, num_steps, team_size, active_agents, num_freeze_steps, reward_scheme=reward_scheme, team_mode=team_mode, seed=seed, agent_type=agent_type
    )
    return env


def make_parallel_envs(args):
    # make parallel environments
    envs = [make_env(args.env_name, args.seed, i, args.num_agents,
                     args.dist_threshold, args.arena_size, args.identity_size) for i in range(args.num_processes)]
    if args.num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(envs)
    else:
        envs = gym_vecenv.DummyVecEnv(envs)

    envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=False, ret=True)
    return envs

# def make_single_env(args):
#     env = make_multiagent_env(args.env_name, args.num_agents, args.dist_threshold, args.arena_size, args.identity_size, args.num_env_steps)
#     return(env)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

from utils import make_open_env
from arguments import get_args
import numpy as np
from learner import Learner
from mpnn import MPNN
from rlagent import Neo

import time
from gym.vector import AsyncVectorEnv
from datetime import date
import string
import random
from torch.utils.tensorboard import SummaryWriter
import os
import json
from Agent import MRFAgent

def prep_obs(obs):
    return np.reshape(obs, (1, obs.shape[0], -1))

if __name__ == "__main__":
    args = vars(get_args())


    def make_env(args, rank, num_agents=5, active_agents=3, freeze_multiplier=80, team_mode="guard",
                 reward_scheme="sparse", seed=100):
        def _init():
            return make_open_env(
                args, args['num_env_steps'], num_agents, active_agents, freeze_multiplier,
                team_mode=team_mode, reward_scheme=reward_scheme, seed=int(seed + 1000 * rank)
            )

        return _init


    num_players_train = args['num_players_train']
    num_players_test = args['num_players_test']

    env = AsyncVectorEnv([
        make_env(args, i, active_agents=num_players_train, seed=args['seed'], reward_scheme="sparse") for i
        in range(8)
    ])

    args["device"] = "cpu"
    writer = None

    for idx in range(101):
        agent = MRFAgent(args=args, writer=writer, added_u_dim = 0)
        load_dir = args['loading_dir'] + str(idx)
        agent.load_parameters(load_dir)

        obs_list = []

        agent.reset()
        obs = env.reset()
        for i in range(3000) :
            print(idx, i)
            obs_list.append(obs)
            acts = agent.step(obs, eval=True)
            n_obs, reward, done, info = env.step(acts)
            obs = n_obs

        agent.save_util_storage("utils_"+str(idx))
        agent.clear_util_storage()
        np.save('obs_'+str(idx)+'.npy', obs_list)

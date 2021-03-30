import random
import numpy as np
from lbforaging.foraging.agent import Agent
from lbforaging.agents.A2CAssets.FCNetwork import FCNetwork
from torch.distributions.categorical import Categorical
import numpy as np
import os
import torch


class A2CAgent(Agent):
    name = "A2CAgent"

    def __init__(self, policy_type):
        self.policy = FCNetwork((24, 96, 64, 6), output_activation=None)
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.load_parameters(current_path+"/A2CAssets/agent_policy_"+str(policy_type)+"_40")

    def step(self, obs):
        obs_tensor = torch.Tensor(obs[:-1]).view(1,-1)
        act_logits = self.policy(obs_tensor)

        act_dist = Categorical(logits=act_logits)
        acts = act_dist.sample().tolist()

        return acts[0]

    def load_parameters(self, param_dir):
        self.policy.load_state_dict(torch.load(param_dir))
        self.policy.eval()
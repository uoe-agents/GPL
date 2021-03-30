import numpy as np
# from baselines.common.vec_env import VecEnv
# from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from baselines.common.vec_env import DummyVecEnv
class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)
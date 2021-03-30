import random
from Wolfpack_gym.envs.wolfpack_assets.ReplayMemory import ReplayMemoryLite
from Wolfpack_gym.envs.wolfpack_assets.QNetwork import DQN
from Wolfpack_gym.envs.wolfpack_assets.misc import hard_copy, soft_copy
import torch
import torch.optim as optim
import numpy as np

class Agent(object):
    def __init__(self, agent_id, obs_type):
        self.agent_id = agent_id
        self.obs_type = obs_type

    def get_obstype(self):
        return self.obs_type

class DQNAgent(Agent):
    def __init__(self, agent_id, args=None, obs_type="partial_obs", obs_height=9, obs_width=17, mode="test"):
        super(DQNAgent, self).__init__(agent_id, obs_type)
        self.obs_type = obs_type
        self.args = args
        self.color = (255, 0, 0)
        self.experience_replay = ReplayMemoryLite(state_h=obs_height, state_w=obs_width,
                                                  with_gpu=self.args['with_gpu'])
        self.dqn_net = DQN(17,9,32,self.args['max_seq_length'],7, mode="partial")

        if self.args['with_gpu']:
            self.dqn_net.cuda()
            self.dqn_net.device = "cuda:0"
            self.target_dqn_net.cuda()
            self.target_dqn_net.device = "cuda:0"

        self.mode = mode
        if not self.mode == "test":
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])
            self.target_dqn_net = DQN(17, 9, 32, self.args['max_seq_length'], 7, mode="partial")
            hard_copy(self.target_dqn_net, self.dqn_net)

        self.recent_obs_storage = np.zeros([self.args['max_seq_length'], obs_height, obs_width, 3])


    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        self.dqn_net.eval()

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)

    def act(self, obs,added_features=None, mode="train", epsilon=0.01):
        self.recent_obs_storage = np.roll(self.recent_obs_storage, axis=0, shift=-1)
        self.recent_obs_storage[-1] = obs
        net_inp = torch.Tensor([self.recent_obs_storage.transpose([0, 3, 1, 2])])
        _, indices = torch.max(self.dqn_net(net_inp), dim=-1)
        # Implement resets
        if not self.mode=="test":
            if random.random() < epsilon:
                indices = random.randint(0,6)
        return indices

    def store_exp(self, exp):
        self.experience_replay.insert(exp)

    def get_obs_type(self):
        return self.obs_type

    def update(self):
        if self.experience_replay.size < self.args['sampling_wait_time']:
            return
        batched_data = self.experience_replay.sample(self.args['batch_size'])
        state, action, reward, dones, next_states = batched_data[0], batched_data[1], batched_data[2], \
                                                    batched_data[3], batched_data[4]

        state = state.permute(0, 1, 4, 2, 3)
        next_states = next_states.permute(0, 1, 4, 2, 3)

        predicted_value = self.dqn_net(state).gather(1, action.long())
        target_values = reward + self.args['disc_rate'] * (1 - dones) * torch.max(self.target_dqn_net(next_states),
                                                                                  dim=-1, keepdim=True)[0]
        loss = 0.5 * torch.mean((predicted_value - target_values.detach()) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_copy(self.target_dqn_net, self.dqn_net)
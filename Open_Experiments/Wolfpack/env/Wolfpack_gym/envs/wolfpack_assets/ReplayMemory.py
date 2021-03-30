from random import sample
import numpy as np
import torch

class ReplayMemoryLite(object):
    def __init__(self, max_capacity=1000, state_h=25, state_w=25, len_seq=5, with_added=False, added_dim=0,
                 with_gpu=False):
        self.max_capacity = max_capacity
        self.state_h = state_h
        self.state_w = state_w
        self.len_seq = int(len_seq)

        self.episode_counter = 0
        self.size = 0
        self.pointer = 0

        self.device = 'cpu'
        if with_gpu:
            self.device = 'cuda:0'

        self.with_added = with_added
        if self.with_added:
            self.added_dim = added_dim

        self.eps_number = np.zeros([int(max_capacity+self.len_seq)])
        self.states = np.zeros([int(max_capacity+self.len_seq), self.state_h, self.state_w, 3])
        self.rewards = np.zeros([int(max_capacity)])
        self.dones = np.zeros([int(max_capacity)])
        self.actions = np.zeros([int(max_capacity)])

        if self.with_added:
            self.added_states = np.zeros([int(max_capacity+self.len_seq), self.added_dim])

    def insert(self, data):
        states, actions, rewards, dones, next_states = data[0], data[1], data[2], data[3], data[4]

        if not self.size < self.max_capacity:
            self.states = np.roll(self.states, axis=0, shift=-1)
            self.eps_number = np.roll(self.eps_number, axis=0, shift=-1)
            self.actions = np.roll(self.actions, axis=0, shift=-1)
            self.dones = np.roll(self.dones, axis=0, shift=-1)
            self.rewards = np.roll(self.rewards, axis=0, shift=-1)
            if self.with_added:
                self.added_states = np.roll(self.added_states, axis=0, shift=-1)

        if not self.with_added:
            self.states[self.pointer+self.len_seq-1] = states
            self.states[self.pointer + self.len_seq] = next_states
        else:
            self.states[self.pointer + self.len_seq - 1] = states[0]
            self.states[self.pointer + self.len_seq] = next_states[0]
            self.added_states[self.pointer + self.len_seq - 1] = states[1]
            self.added_states[self.pointer + self.len_seq] = next_states[1]

        self.eps_number[self.pointer+self.len_seq-1] = self.episode_counter
        self.actions[self.pointer] = actions
        self.dones[self.pointer] = int(dones)
        self.rewards[self.pointer] = rewards
        self.eps_number[self.pointer+self.len_seq] = self.episode_counter

        self.pointer = min(self.pointer + 1, self.max_capacity-1)
        self.size = min(self.size+1, self.max_capacity)

        if int(dones):
            self.episode_counter += 1

    def sample(self,batch_size, idxes=None):
        if batch_size < self.size:
            sampled_idx = sample(range(self.size), batch_size)
            if idxes != None:
                sampled_idx = idxes
            shifted_idxes = [[a + self.len_seq - 1 - x for a in sampled_idx] for x in reversed(range(self.len_seq))]
            if self.with_added:
                added_seq_data = np.asarray([self.added_states[indices] for indices in shifted_idxes]).swapaxes(0,1)
            # Swap time and batch
            seq_data = np.asarray([self.states[indices] for indices in shifted_idxes]).swapaxes(0,1)
            filter_flags = [self.eps_number[idx:idx+self.len_seq] ==
                             self.eps_number[idx+self.len_seq-1] for idx in sampled_idx]
            # Make masks for zeros
            if self.with_added:
                added_seq_data_used = torch.Tensor(np.asarray([[c*d for c,d in zip(a,b)]for a,b in
                                                               zip(added_seq_data, filter_flags)])).to(self.device)
            seq_data_used = torch.Tensor(np.asarray([[c*d for c,d in zip(a,b)]for a,b in
                                                     zip(seq_data, filter_flags)])).to(self.device)

            #next_data
            next_idxes = [[a + self.len_seq - 1 - x for a in sampled_idx] for x in reversed(range(-1,self.len_seq-1))]
            if self.with_added:
                added_seq_next_data = np.asarray([self.added_states[indices]
                                                  for indices in next_idxes]).swapaxes(0, 1)
            next_seq_data = np.asarray([self.states[indices] for indices in next_idxes]).swapaxes(0, 1)
            next_filter_flags = [self.eps_number[idx+1:idx + self.len_seq+1] ==
                            self.eps_number[idx + self.len_seq] for idx in sampled_idx]

            if self.with_added:
                added_next_seq_data_used = torch.Tensor(np.asarray([[c * d for c, d in zip(a, b)] for a, b in
                                                          zip(added_seq_next_data,
                                                              next_filter_flags)])).to(self.device)
            next_seq_data_used = torch.Tensor(np.asarray([[c * d for c, d in zip(a, b)] for a, b in
                                                          zip(next_seq_data,
                                                              next_filter_flags)])).to(self.device)
            actions_batch = torch.Tensor(self.actions[sampled_idx]).to(self.device).unsqueeze(-1)
            dones_batch = torch.Tensor(self.dones[sampled_idx]).to(self.device).unsqueeze(-1)
            rews_batch = torch.Tensor(self.rewards[sampled_idx]).to(self.device).unsqueeze(-1)

            if self.with_added:
                return seq_data_used, actions_batch, rews_batch, dones_batch, next_seq_data_used

            return seq_data_used, added_seq_data_used, actions_batch, rews_batch, \
                   dones_batch, next_seq_data_used, added_next_seq_data_used
        return
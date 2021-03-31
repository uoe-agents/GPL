import math

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import Variable
import numpy as np
import dgl

def hard_copy(target_net, source_net):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)

def soft_copy(target_net, source_net, tau=0.001):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau*param + (1-tau)*target_param)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_tensor = (logits == logits.max(1, keepdim=True)[0]).float()
    random_tensor = torch.eye(logits.shape[1])[[np.random.choice(logits.shape[1], size=logits.shape[0])]]

    return torch.stack(
        [argmax_tensor[i] if samp > eps else random_tensor[i] for i, samp in enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, graph, eps=1e-20, tens_type=torch.FloatTensor):
    # Get gumberl noise
    noises = []
    for unbatched_graph in dgl.unbatch(graph):
        num_node = unbatched_graph.number_of_nodes()
        src, dst = unbatched_graph.edges()
        a1, a2 = tens_type(*[num_node,num_node]).uniform_(), tens_type(*[num_node,num_node]).uniform_()
        tri_tensor1, tri_tensor2 = torch.tril(a1,-1) + torch.tril(a1,-1).permute(1,0), \
                                   torch.tril(a2,-1) + torch.tril(a2,-1).permute(1,0)
        noise = torch.cat([tri_tensor1[src, dst][:,None], tri_tensor2[src, dst][:,None]], dim=-1)
        noises.append(noise)

    U = Variable(torch.cat(noises, dim=0), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, graph, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, graph, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

def categorical_sample(logits, graph):

    noises = []
    offset = 0
    for unbatched_graph in dgl.unbatch(graph):
        num_node = unbatched_graph.number_of_nodes()
        src, dst = unbatched_graph.edges()
        sampled_indexes = torch.zeros([num_node, num_node]).long()
        sampled_logits = logits[offset:offset+(num_node*(num_node-1)),:]
        sample = dist.Categorical(logits=sampled_logits).sample()
        sampled_indexes[src,dst] = sample

        src, dst = unbatched_graph.edges()
        tri_tensor1 = (torch.tril(sampled_indexes, -1) + torch.tril(sampled_indexes, -1).permute(1, 0))
        noise = tri_tensor1[src, dst]
        noise_sample = torch.eye(logits.shape[1])[[noise]]
        noises.append(noise_sample)
        offset += num_node*(num_node-1)

    categorical_samples = torch.cat(noises, dim=0)
    return categorical_samples

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, graph, temperature=0.5, hard=False, epsilon=0.0, evaluate=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, graph, temperature)

    if evaluate:
        #return onehot_from_logits(logits, eps=epsilon)
        # y_onehot = torch.FloatTensor(logits.shape[0], 2)
        # # In your for loop
        # y_onehot.zero_()
        # y = dist.Categorical(logits=logits).sample()[:,None]
        # y_onehot.scatter_(1, y, 1)

        y_onehot = onehot_from_logits(logits, eps=0.0)
        return y_onehot

    if hard:
        y_hard = onehot_from_logits(y, eps=epsilon)
        y = (y_hard - y).detach() + y
    return y

class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

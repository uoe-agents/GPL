import random
from Network import *
import torch
import torch.nn as nn
import torch.optim as optim
import dgl


# Add storage for the neighbourhood flags

def hard_copy(target_net, source_net):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)

def soft_copy(target_net, source_net, tau=0.001):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau*param + (1-tau)*target_param)

class MRFAgent(object):
    def __init__(self, args=None, optimizer=None, device=None, writer=None,
                 epsilon=1.0, added_u_dim=0, mode="train", gumbel_temp=None):

        self.args = args
        self.added_u_dim = added_u_dim

        self.pair_comp = self.args['pair_comp']
        self.writer = writer
        self.num_updates = 0

        # Initialize neural network dimensions
        self.dim_lstm_out = 128
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dqn_net = LSTMMRF(7, self.added_u_dim, 64, 128, 64, 128,
                                      8, 70, 40, 128, pair_comp=self.pair_comp).to(self.device)
        self.target_dqn_net = LSTMMRF(7, self.added_u_dim, 64, 128, 64, 128,
                                            8, 70, 40, 128, pair_comp=self.pair_comp).to(self.device)

        hard_copy(self.target_dqn_net, self.dqn_net)
        self.mode = mode

        # Initialize hidden states for prediction
        self.hiddens = None
        self.hiddens_u = None
        self.prev_hiddens = None
        self.prev_hiddens_u = None
        self.target_hiddens = None
        self.target_hiddens_u = None

        # Set params for Ad Hoc BPTT
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])

        self.targ_vals = []
        self.pred_vals = []
        self.logit_probs_pi = []
        self.logit_probs_theta = []
        self.joint_actions = []

        self.epsilon = epsilon
        self.loss_module = nn.MSELoss()
        self.loss_module_f = nn.CrossEntropyLoss()

    def step(self, obs, eval=False):

        p_graph, n_ob, u_ob, n_hiddens, n_hiddens_u = self.prep(
            obs, self.hiddens, self.hiddens_u
        )

        out, log_prob_theta, hids, hids_u = self.dqn_net(
            p_graph, n_ob, n_ob, u_ob, n_hiddens, n_hiddens_u
        )

        self.prev_hiddens = self.hiddens
        self.prev_hiddens_u = self.hiddens_u
        self.hiddens = hids
        self.hiddens_u = hids_u

        acts = torch.argmax(out, dim=-1).tolist()

        # Select outputs so likelihood only computed for nodes other than agent
        zero_indexes, offset = [0], 0
        num_nodes = p_graph.batch_num_nodes
        all_num_nodes = sum(num_nodes)

        for a in num_nodes[:-1]:
            offset += a
            zero_indexes.append(offset)

        non_zero_indices = torch.Tensor([k for k in range(all_num_nodes) if not (k in zero_indexes)]).long()

        if not eval:
            acts = [a if random.random() > self.epsilon else random.randint(0, 7) for a in acts]
            self.logit_probs_theta.append(log_prob_theta[non_zero_indices,:])

        return acts

    def compute_target(self, obs, acts, rewards, done, n_obs, add_storage=True):

        if add_storage:
            prev_p_graph, prev_n_ob, prev_u_ob, prev_n_hiddens, prev_n_hiddens_u = self.prep(
                obs, self.prev_hiddens, self.prev_hiddens_u
            )

        prep_outs = self.prep(
            n_obs, self.target_hiddens, self.target_hiddens_u, with_acts=add_storage, add_acts=acts
        )

        target_out, _, target_hids, target_hids_u = self.target_dqn_net(
            prep_outs[0], prep_outs[1], prep_outs[1],
            prep_outs[2], prep_outs[3], prep_outs[4]
        )

        self.target_hiddens = target_hids
        self.target_hiddens_u = target_hids_u

        if add_storage:
            out = self.dqn_net(
                prev_p_graph, prev_n_ob, prev_n_ob, prev_u_ob,
                prev_n_hiddens, prev_n_hiddens_u,
                mrf_mode="joint", joint_acts=prep_outs[5]
            )

            # Select outputs so pseudolikelihood only computed for nodes other than agent
            zero_indexes, offset = [0], 0
            num_nodes = prev_p_graph.batch_num_nodes
            all_num_nodes = sum(num_nodes)

            for a in num_nodes[:-1]:
                offset += a
                zero_indexes.append(offset)

            non_zero_indices_list = [k for k in range(all_num_nodes) if not (k in zero_indexes)]
            non_zero_indices = torch.Tensor(non_zero_indices_list).long()

            self.joint_actions.extend([x for idx, x in enumerate(prep_outs[5]) if idx in non_zero_indices_list])
            self.logit_probs_pi.append(out[1][non_zero_indices,:])
            self.pred_vals.append(out[0])

            rew_t = torch.Tensor(rewards)[:, None].to(self.device)
            dones = torch.Tensor(done)[:, None].to(self.device)
            targets = rew_t + self.args['gamma'] * (1 - dones) * torch.max(target_out, dim=-1, keepdim=True)[0]
            self.targ_vals.append(targets)

    def detach_hiddens(self):
        self.hiddens = (self.hiddens[0].detach(), self.hiddens[1].detach())
        self.hiddens_u = (self.hiddens_u[0].detach(), self.hiddens_u[1].detach())
 
    def prep(self, obs, hiddens, hiddens_u, with_acts=False, add_acts=None):
        graph_list = []
        num_agents = (obs[:,:,0] != 0).sum(axis=-1)
        prev_existing_agents_data = obs[obs[:,:,-1] != -1]
        cur_existing_agents_data = obs[obs[:,:,0] != 0]

        prev_existing_agents_and_alive_flag = prev_existing_agents_data[:, 0] != 0
        cur_existing_agents_and_prev_existing_flag = cur_existing_agents_data[:, -1] != -1

        # Create graph
        for num_agent in num_agents:
            num_agent = int(num_agent)
            graph_ob = dgl.DGLGraph()
            graph_ob.add_nodes(num_agent)
            if num_agent > 1:
                src, dst = zip(*[(a, b) for a in range(num_agent) for b in range(num_agent) if a != b])
                graph_ob.add_edges(src, dst)
            graph_list.append(graph_ob)

        graph_batch = dgl.batch(graph_list)

        # Parse inputs into node inputs

        num_nodes = graph_batch.batch_num_nodes
        n_ob = torch.Tensor(cur_existing_agents_data[:, 1:-1]).float()
        u_ob = torch.zeros((len(num_agents),0)).float()

        # Create filters to decide which hidden vectors to maintain
        # For newly added agents, hiddens set to zeros
        # For remaining agents, hiddens continues from prev timestep

        node_filter = torch.tensor(prev_existing_agents_and_alive_flag, dtype=torch.bool)
        complete_new_filter = torch.tensor(cur_existing_agents_and_prev_existing_flag, dtype=torch.bool)

        # Create action vectors for opponent modelling
        if with_acts:
            acts = prev_existing_agents_data[:,-1].astype(int).tolist()

        # Filter hidden vectors for remaining agents
        # Add zero vectors for newly added agents
        n_hid = (torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out]),
                 torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out]))

        if not (hiddens is None) and node_filter.size != 0:
            collected_hiddens = (hiddens[0][:,node_filter,:], hiddens[1][:,node_filter,:])
            n_hid[0][:, complete_new_filter, :] = collected_hiddens[0]
            n_hid[1][:, complete_new_filter, :] = collected_hiddens[1]

        n_hid_u = (torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out]),
                     torch.zeros([1,graph_batch.number_of_nodes(), self.dim_lstm_out]))

        if not (hiddens_u is None) and node_filter.size != 0:
            collected_hiddens = (hiddens_u[0][:, node_filter, :], hiddens_u[1][:, node_filter, :])
            n_hid_u[0][:, complete_new_filter, :] = collected_hiddens[0]
            n_hid_u[1][:, complete_new_filter, :] = collected_hiddens[1]

        if with_acts:
            return graph_batch, n_ob, u_ob, n_hid, n_hid_u, acts

        return graph_batch, n_ob, u_ob, n_hid, n_hid_u

    def reset(self):
        self.hiddens = None
        self.hiddens_u = None
        self.target_hiddens = None
        self.target_hiddens_u = None

        self.targ_vals = []
        self.pred_vals = []
        self.logit_probs_pi = []
        self.logit_probs_theta = []
        self.joint_actions = []

    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename))
        self.target_dqn_net.state_dict(torch.load(filename + "_target_dqn"))

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)
        torch.save(self.target_dqn_net.state_dict(), filename + "_target_dqn")

    def set_epsilon(self, eps):
        self.epsilon = eps

    def update(self):
        self.optimizer.zero_grad()

        # Util losses
        pred_tensor = torch.cat(self.pred_vals, dim=0)
        target_tensor = torch.cat(self.targ_vals, dim=0)

        # modelling losses
        joint_actions = torch.Tensor(self.joint_actions).long()
        joint_logit_theta = torch.cat(self.logit_probs_theta, dim=0)
        theta_cross_entropy = self.loss_module_f(joint_logit_theta, joint_actions)

        loss_pred = theta_cross_entropy
        val_loss = self.loss_module(pred_tensor, target_tensor.detach())

        loss = val_loss + self.args['weight_predict'] * loss_pred

        self.writer.add_scalar('loss/q_loss', val_loss, self.num_updates)
        self.writer.add_scalar('loss/theta_cross_entropy', theta_cross_entropy, self.num_updates)

        loss.backward()
        self.optimizer.step()

        soft_copy(self.target_dqn_net, self.dqn_net, self.args['tau'])
        self.target_dqn_net.hard_copy_fs(self.dqn_net)

        self.targ_vals = []
        self.pred_vals = []
        self.logit_probs_pi = []
        self.logit_probs_theta = []
        self.joint_actions = []

        self.detach_hiddens()

        self.num_updates += 1
        

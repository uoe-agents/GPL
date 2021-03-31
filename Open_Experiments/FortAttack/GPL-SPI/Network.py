import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch.distributions as dist
import dgl
import numpy as np

class RFMBlock(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, hidden_dim, dim_out):
        super(RFMBlock, self).__init__()
        self.fc_edge = nn.Linear(dim_in_edge,hidden_dim)
        self.fc_edge2 = nn.Linear(hidden_dim, dim_out)
        self.fc_node = nn.Linear(dim_in_node, hidden_dim)
        self.fc_node2 = nn.Linear(hidden_dim, dim_out)
        self.fc_u = nn.Linear(dim_in_u, hidden_dim)
        self.fc_u2 = nn.Linear(hidden_dim, dim_out)
        # Check Graph batch

        self.graph_msg = fn.copy_edge(edge='edge_feat', out='m')
        self.graph_reduce = fn.sum(msg='m', out='h')

    def graph_message_func(self,edges):
        return {'m': edges.data['edge_feat'] }

    def graph_reduce_func(self,nodes):
        msgs = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': msgs}

    def compute_edge_repr(self, graph, edges, g_repr):
        edge_nums = graph.batch_num_edges
        u = torch.cat([g[None,:].repeat(num_edge,1) for g, num_edge
                       in zip(g_repr,edge_nums) if num_edge != 0], dim=0)
        inp = torch.cat([edges.data['edge_feat'],edges.src['node_feat'],edges.dst['node_feat'], u], dim=-1)
        return {'edge_feat' : self.fc_edge2(F.relu(self.fc_edge(inp)))}

    def compute_node_repr(self, graph, nodes, g_repr):
        node_nums = graph.batch_num_nodes
        u = torch.cat([g[None, :].repeat(num_node, 1) for g, num_node
                       in zip(g_repr, node_nums)], dim=0)
        inp = torch.cat([nodes.data['node_feat'], nodes.data['h'], u], dim=-1)
        return {'node_feat' : self.fc_node2(F.relu(self.fc_node(inp)))}

    def compute_u_repr(self, n_comb, e_comb, g_repr):
        inp = torch.cat([n_comb, e_comb, g_repr], dim=-1)
        return self.fc_u2(F.relu(self.fc_u(inp)))

    def forward(self, graph, edge_feat, node_feat, g_repr):
        node_trf_func = lambda x: self.compute_node_repr(nodes=x, graph=graph, g_repr=g_repr)

        graph.edata['edge_feat'] = edge_feat
        graph.ndata['node_feat'] = node_feat
        edge_trf_func = lambda x : self.compute_edge_repr(edges=x, graph=graph, g_repr=g_repr)

        graph.apply_edges(edge_trf_func)
        graph.update_all(self.graph_message_func, self.graph_reduce_func, node_trf_func)

        e_comb = dgl.sum_edges(graph, 'edge_feat')
        n_comb = dgl.sum_nodes(graph, 'node_feat')

        e_out = graph.edata['edge_feat']
        n_out = graph.ndata['node_feat']

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())
        for key in e_keys:
            graph.edata.pop(key)
        for key in n_keys:
            graph.ndata.pop(key)

        return e_out, n_out, self.compute_u_repr(n_comb, e_comb, g_repr)

class RFMBlockPseudolikelihood(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, hidden_dim, dim_out, act_dims):
        super(RFMBlockPseudolikelihood, self).__init__()
        self.fc_edge = nn.Linear(dim_in_edge+act_dims,hidden_dim)
        self.fc_edge2 = nn.Linear(hidden_dim, dim_out)
        self.fc_node = nn.Linear(dim_in_node, hidden_dim)
        self.fc_node2 = nn.Linear(hidden_dim, dim_out)
        self.fc_u = nn.Linear(dim_in_u, hidden_dim)
        self.fc_u2 = nn.Linear(hidden_dim, dim_out)
        # Check Graph batch

        self.graph_msg = fn.copy_edge(edge='edge_feat', out='m')
        self.graph_reduce = fn.sum(msg='m', out='h')

    def graph_message_func(self,edges):
        return {'m': edges.data['edge_feat'] }

    def graph_reduce_func(self,nodes):
        msgs = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': msgs}

    def compute_edge_repr(self, graph, edges, g_repr):
        edge_nums = graph.batch_num_edges
        u = torch.cat([g[None,:].repeat(num_edge,1) for g, num_edge
                       in zip(g_repr,edge_nums) if num_edge != 0], dim=0)
        inp = torch.cat([edges.data['edge_feat'],edges.src['node_feat'], edges.src['add_acts'],
                         edges.dst['node_feat'], u], dim=-1)
        return {'edge_feat' : self.fc_edge2(F.relu(self.fc_edge(inp)))}

    def compute_node_repr(self, graph, nodes, g_repr):
        node_nums = graph.batch_num_nodes
        u = torch.cat([g[None, :].repeat(num_node, 1) for g, num_node
                       in zip(g_repr, node_nums)], dim=0)
        inp = torch.cat([nodes.data['node_feat'], nodes.data['h'], u], dim=-1)
        return {'node_feat' : self.fc_node2(F.relu(self.fc_node(inp)))}

    def compute_u_repr(self, n_comb, e_comb, g_repr):
        inp = torch.cat([n_comb, e_comb, g_repr], dim=-1)
        return self.fc_u2(F.relu(self.fc_u(inp)))

    def forward(self, graph, edge_feat, node_feat, g_repr, add_acts):
        node_trf_func = lambda x: self.compute_node_repr(nodes=x, graph=graph, g_repr=g_repr)

        graph.edata['edge_feat'] = edge_feat
        graph.ndata['node_feat'] = node_feat
        graph.ndata['add_acts'] = add_acts
        edge_trf_func = lambda x : self.compute_edge_repr(edges=x, graph=graph, g_repr=g_repr)

        graph.apply_edges(edge_trf_func)
        graph.update_all(self.graph_message_func, self.graph_reduce_func, node_trf_func)

        e_comb = dgl.sum_edges(graph, 'edge_feat')
        n_comb = dgl.sum_nodes(graph, 'node_feat')

        e_out = graph.edata['edge_feat']
        n_out = graph.ndata['node_feat']

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())
        for key in e_keys:
            graph.edata.pop(key)
        for key in n_keys:
            graph.ndata.pop(key)

        return e_out, n_out, self.compute_u_repr(n_comb, e_comb, g_repr)

class UtilLayer(nn.Module):
    def __init__(self, dim_in_node, mid_pair, mid_nodes, num_acts,
                 pair_comp="avg", mid_pair_out=8, device=None):
        super(UtilLayer, self).__init__()
        self.pair_comp = pair_comp
        self.mid_pair = mid_pair
        self.num_acts = num_acts
        self.utils_storage = {}

        self.utils_storage["indiv"] = []
        self.utils_storage["pairs"] = []
        self.utils_storage["batch_num_nodes"] = []
        self.utils_storage["batch_num_edges"] = []

        if self.pair_comp=="bmm":
            self.ju1 = nn.Linear(3*dim_in_node, self.mid_pair)
            self.ju3 = nn.Linear(self.mid_pair, self.mid_pair)
        else:
            self.ju3 = nn.Linear(self.mid_pair, self.mid_pair)
            self.ju1 = nn.Linear(3*dim_in_node, self.mid_pair)

        if self.pair_comp=="bmm":
            self.mid_pair_out = mid_pair_out
            self.ju2 = nn.Linear(self.mid_pair,num_acts*self.mid_pair_out)
        else:
            self.ju2 = nn.Linear(self.mid_pair, num_acts * num_acts)

        self.iu1 = nn.Linear(2*dim_in_node, mid_nodes)
        self.iu3 = nn.Linear(mid_nodes, mid_nodes)
        self.iu2 = nn.Linear(mid_nodes, num_acts)

        self.num_acts = num_acts
        self.device = device

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def compute_node_data(self, nodes):
        return {'indiv_util': self.iu2(F.relu(self.iu3(F.relu(self.iu1(nodes.data['node_feat_u'])))))}

    def compute_edge_data(self, edges):
        inp_u = edges.data['edge_feat_u']
        inp_reflected_u = edges.data['edge_feat_reflected_u']

        if self.pair_comp == 'bmm':
            # Compute the util components
            util_comp = self.ju2(F.relu(self.ju3(F.relu(self.ju1(inp_u))))).view(-1,self.num_acts, self.mid_pair_out)
            util_comp_reflected = self.ju2(F.relu(self.ju3(F.relu(self.ju1(inp_reflected_u))))).view(-1,self.num_acts,
                                                                                   self.mid_pair_out).permute(0,2,1)

            util_vals = torch.bmm(util_comp, util_comp_reflected).permute(0,2,1)
        else:
            util_comp = self.ju2(F.relu(self.ju3(F.relu(self.ju1(inp_u))))).view(-1, self.num_acts, self.num_acts)
            util_comp_reflected = self.ju2(F.relu(self.ju3(F.relu(self.ju1(inp_reflected_u))))).view(-1, self.num_acts,
                                                                                 self.num_acts).permute(0,2,1)

            util_vals = ((util_comp + util_comp_reflected)/2.0).permute(0,2,1)

        final_u_factor = util_vals
        reflected_util_vals = final_u_factor.permute(0, 2, 1)

        return {'util_vals': final_u_factor,
                'reflected_util_vals': reflected_util_vals}

    def graph_pair_inference_func(self, edges):
        src_prob, dst_prob = edges.src["probs"], edges.dst["probs"]
        edge_all_sum = (edges.data["util_vals"] * src_prob.unsqueeze(1) *
                        dst_prob.unsqueeze(-1)).sum(dim=-1).sum(dim=-1,
                        keepdim=True)
        return {'edge_all_sum_prob': edge_all_sum}

    def graph_dst_inference_func(self, edges):
        src_prob = edges.src["probs"]
        u_message = (edges.data["util_vals"] * src_prob.unsqueeze(1)).sum(dim=-1)

        return {'marginalized_u' : u_message}

    def graph_node_inference_func(self, nodes):
        indiv_util = nodes.data["indiv_util"]
        weighting = nodes.data["probs"]

        return {"expected_indiv_util" : (indiv_util*weighting).sum(dim=-1)}

    def graph_reduce_func(self, nodes):
        util_msg = torch.sum(nodes.mailbox['marginalized_u'], dim=1)
        return {'util_dst': util_msg}

    def graph_u_sum(self, graph, edges, acts):
        src, dst = graph.edges()
        acts_src = torch.Tensor([acts[idx] for idx in src.tolist()])

        u = edges.data['util_vals']
        reshaped_acts = acts_src.view(u.shape[0], 1, -1).long().repeat(1, self.num_acts, 1)
        u_msg = u.gather(-1, reshaped_acts).permute(0,2,1).squeeze(1)
        return {'u_msg': u_msg}

    def graph_sum_all(self, nodes):
        util_msg = torch.sum(nodes.mailbox['u_msg'], dim=1)
        return {'u_msg_sum': util_msg}

    def clear_util_storage(self):
        self.utils_storage["indiv"] =[]
        self.utils_storage["pairs"] = []
        self.utils_storage["batch_num_nodes"] = []
        self.utils_storage["batch_num_edges"] = []


    def save_util_storage(self, filename):
        np.save(filename+"_pairs.npy", self.utils_storage["pairs"])
        np.save(filename+"_indiv.npy", self.utils_storage["indiv"])
        np.save(filename + "_batch_num_nodes.npy", self.utils_storage["batch_num_nodes"])
        np.save(filename + "_batch_num_edges.npy", self.utils_storage["batch_num_edges"])

    def forward(self, graph, edge_feats_u, node_feats_u,
                edge_feat_reflected_u, mode="train",
                node_probability = None,
                joint_acts=None):

        graph.edata['edge_feat_u'] = edge_feats_u
        graph.edata['edge_feat_reflected_u'] = edge_feat_reflected_u
        graph.ndata['node_feat_u'] = node_feats_u

        n_weights = torch.zeros([node_feats_u.shape[0],1])

        zero_indexes, offset = [0], 0
        num_nodes = graph.batch_num_nodes

        # Mark all 0-th index nodes
        for a in num_nodes[:-1]:
            offset += a
            zero_indexes.append(offset)

        n_weights[zero_indexes] = 1
        graph.ndata['weights'] = n_weights
        graph.ndata['mod_weights'] = 1-n_weights

        graph.apply_nodes(self.compute_node_data)
        graph.apply_edges(self.compute_edge_data)

        self.utils_storage["indiv"].append(graph.ndata["indiv_util"].detach().numpy())
        self.utils_storage["pairs"].append(graph.edata["util_vals"].detach().numpy())
        self.utils_storage["batch_num_nodes"].append(graph.batch_num_nodes)
        self.utils_storage["batch_num_edges"].append(graph.batch_num_edges)

        if "inference" in mode:
            graph.ndata["probs"] = node_probability
            src, dst = graph.edges()
            src_list, dst_list = src.tolist(), dst.tolist()

            # Mark edges not connected to zero
            e_nc_zero_weight = torch.zeros([edge_feats_u.shape[0],1])
            all_nc_edges = [idx for idx, (src, dst) in enumerate(zip(src_list,dst_list)) if
                            (not src in zero_indexes) and (not dst in zero_indexes)]
            e_nc_zero_weight[all_nc_edges] = 0.5
            graph.edata["nc_zero_weight"] = e_nc_zero_weight

            graph.apply_edges(self.graph_pair_inference_func)
            graph.update_all(message_func=self.graph_dst_inference_func, reduce_func=self.graph_reduce_func,
                             apply_node_func=self.graph_node_inference_func)

            total_connected = dgl.sum_nodes(graph, 'util_dst', 'weights')
            total_n_connected = dgl.sum_edges(graph, 'edge_all_sum_prob', 'nc_zero_weight')
            total_expected_others_util = dgl.sum_nodes(graph, "expected_indiv_util", "mod_weights").view(-1,1)
            total_indiv_util_zero = dgl.sum_nodes(graph, "indiv_util", "weights")

            returned_values = (total_connected + total_n_connected) + \
                              (total_expected_others_util + total_indiv_util_zero)

            e_keys = list(graph.edata.keys())
            n_keys = list(graph.ndata.keys())

            for key in e_keys:
                graph.edata.pop(key)

            for key in n_keys:
                graph.ndata.pop(key)

            return returned_values

        m_func = lambda x: self.graph_u_sum(graph, x, joint_acts)
        graph.update_all(message_func=m_func,
                        reduce_func=self.graph_sum_all)

        indiv_u_zeros = graph.ndata['indiv_util']
        u_msg_sum_zeros = 0.5 * graph.ndata['u_msg_sum']

        graph.ndata['utils_sum_all'] = (indiv_u_zeros + u_msg_sum_zeros).gather(-1,
                                                                                torch.Tensor(joint_acts)[:,None].long())
        q_values = dgl.sum_nodes(graph, 'utils_sum_all')

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())

        for key in e_keys:
            graph.edata.pop(key)

        for key in n_keys:
            graph.ndata.pop(key)

        return q_values

class OppoModelNet(nn.Module):
    def __init__(self, dim_in_node, dim_in_u, hidden_dim,
                 dim_lstm_out, dim_mid, dim_out, act_dims,
                 dim_last, rfm_hidden_dim, last_hidden):
        super(OppoModelNet, self).__init__()
        self.dim_lstm_out = dim_lstm_out
        self.act_dims = act_dims

        self.mlp1a = nn.Linear(dim_in_node + dim_in_u, hidden_dim)
        self.mlp1b = nn.Linear(hidden_dim, dim_mid)
        self.lstm1 = nn.LSTM(dim_mid, dim_lstm_out, batch_first=True)
        self.mlp1 = nn.Linear(dim_lstm_out, dim_out)

        self.mlp1_readout = nn.Linear(dim_last, last_hidden)
        self.mlp1_readout2 = nn.Linear(last_hidden, act_dims)

        self.mlp2a = nn.Linear(dim_in_node + dim_in_u, hidden_dim)
        self.mlp2b = nn.Linear(hidden_dim, dim_mid)
        self.lstm2 = nn.LSTM(dim_mid, dim_lstm_out, batch_first=True)
        self.mlp2 = nn.Linear(dim_lstm_out, dim_out)

        self.mlp2_readout = nn.Linear(dim_last, last_hidden)
        self.mlp2_readout2 = nn.Linear(last_hidden, act_dims)

        self.GNBlock_theta = RFMBlock(dim_last+dim_out, 2*dim_out, 2*dim_last, rfm_hidden_dim,
                                 dim_last)
        self.GNBlock_pi = RFMBlockPseudolikelihood(dim_last+dim_out, 2*dim_out, 2*dim_last, rfm_hidden_dim,
                                 dim_last, act_dims)

    def forward(self, graph, obs, hidden_n, mode="theta", add_acts=None):

        if mode == "theta":
            updated_n_feat = self.mlp1b(F.relu(self.mlp1a(obs)))
            updated_n_feat, n_hid = self.lstm1(updated_n_feat.view(updated_n_feat.shape[0], 1, -1), hidden_n)
            updated_n_feat = self.mlp1(F.relu(updated_n_feat.squeeze(1)))

            edge_feat = torch.zeros([graph.number_of_edges(), 0])
            g_repr = torch.zeros([len(graph.batch_num_nodes), 0])

            updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock_theta.forward(graph, edge_feat,
                                                                                  updated_n_feat, g_repr)

            return self.mlp1_readout2(F.relu(self.mlp1_readout(updated_n_feat))), n_hid

        updated_n_feat = self.mlp2b(F.relu(self.mlp2a(obs)))
        updated_n_feat, n_hid = self.lstm2(updated_n_feat.view(updated_n_feat.shape[0], 1, -1), hidden_n)
        updated_n_feat = self.mlp2(F.relu(updated_n_feat.squeeze(1)))

        edge_feat = torch.zeros([graph.number_of_edges(), 0])
        g_repr = torch.zeros([len(graph.batch_num_nodes), 0])

        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock_pi.forward(graph, edge_feat,
                                                                                 updated_n_feat, g_repr,
                                                                                 add_acts)

        return self.mlp2_readout2(F.relu(self.mlp2_readout(updated_n_feat))), n_hid


class LSTMMRF(nn.Module):
    def __init__(self, dim_in_node, dim_in_u, hidden_dim, dim_lstm_out,
                 dim_mid, dim_out, act_dims, dim_last, f_rfm_hidden_dim,
                 f_last_hidden, mid_pair=128, mid_nodes=128, pair_comp="avg",
                 mid_pair_out=6):
        super(LSTMMRF, self).__init__()
        self.dim_lstm_out = dim_lstm_out
        self.act_dims = act_dims

        self.u_mlp1a = nn.Linear(dim_in_node + dim_in_u, hidden_dim)
        self.u_mlp1b = nn.Linear(hidden_dim, dim_mid)
        self.u_lstm2 = nn.LSTM(dim_mid, dim_lstm_out, batch_first=True)
        self.u_mlp2 = nn.Linear(dim_lstm_out, dim_out)

        self.q_net = UtilLayer(dim_out, mid_pair, mid_nodes,
                               act_dims, mid_pair_out=mid_pair_out,
                               pair_comp=pair_comp)

        self.oppo_model_net = OppoModelNet(
            dim_in_node, dim_in_u, hidden_dim, dim_lstm_out,
            dim_mid, dim_out, act_dims,
            dim_last, f_rfm_hidden_dim, f_last_hidden
        )

    def clear_util_storage(self):
        self.q_net.clear_util_storage()

    def save_util_storage(self, filename):
        self.q_net.save_util_storage(filename)

    def forward(self, graph, node_feat, node_feat_u,
                g_repr, hidden_n, hidden_n_u,
                mrf_mode="inference", joint_acts=None):

        u_obs = g_repr
        batch_num_nodes = graph.batch_num_nodes
        add_obs = torch.cat([feat.view(1, -1).repeat(r_num, 1) for
                             feat, r_num in zip(u_obs, batch_num_nodes)], dim=0)
        obs = torch.cat([node_feat, add_obs], dim=-1)

        updated_n_feat_u = self.u_mlp1b(F.relu(self.u_mlp1a(obs)))
        updated_n_feat_u, n_hid_u = self.u_lstm2(updated_n_feat_u.view(updated_n_feat_u.shape[0], 1, -1),
                                                 hidden_n_u)
        updated_n_feat_u_half = self.u_mlp2(F.relu(updated_n_feat_u.squeeze(1)))

        first_elements = [0]
        offset = 0
        for a in batch_num_nodes[:-1]:
            offset += a
            first_elements.append(offset)

        first_elements_u = updated_n_feat_u_half[first_elements, :]
        add_first_elements = torch.cat([feat.view(1, -1).repeat(r_num, 1) for
                                        feat, r_num in zip(first_elements_u, batch_num_nodes)], dim=0)
        updated_n_feat_u = torch.cat([updated_n_feat_u_half, add_first_elements], dim=-1)

        edges = graph.edges()
        e_feat_u_src = updated_n_feat_u_half[edges[0]]
        e_feat_u_dst = updated_n_feat_u_half[edges[1]]

        batch_num_edges = graph.batch_num_edges
        add_first_elements_edge = torch.cat([feat.view(1, -1).repeat(r_num, 1) for
                                             feat, r_num in zip(first_elements_u, batch_num_edges) if r_num !=0], dim=0)

        updated_e_feat_u = torch.cat([e_feat_u_src, e_feat_u_dst, add_first_elements_edge], dim=-1)
        reverse_feats_u = torch.cat([e_feat_u_dst, e_feat_u_src, add_first_elements_edge], dim=-1)

        if "inference" in mrf_mode:

            act_logits, model_hid = self.oppo_model_net(graph, obs, hidden_n)
            node_probs = dist.Categorical(logits=act_logits).probs

            out = self.q_net(
                graph, updated_e_feat_u, updated_n_feat_u, reverse_feats_u,
                mode=mrf_mode, node_probability=node_probs.detach(), joint_acts=joint_acts
            )

            return out, act_logits, model_hid, n_hid_u

        else:
            add_acts = torch.eye(self.act_dims)[torch.Tensor(joint_acts).long(),:]
            act_logits, model_hid = self.oppo_model_net(graph, obs, hidden_n, mode="pi", add_acts=add_acts)


            out = self.q_net(
                graph, updated_e_feat_u,
                updated_n_feat_u, reverse_feats_u,
                mode=mrf_mode, joint_acts=joint_acts
            )

            return out, act_logits

    def hard_copy_fs(self, source):
        for (k, l), (m, n), in zip(self.named_parameters(), source.named_parameters()):
            if ('oppo_model_net' in k):
                l.data.copy_(n.data)

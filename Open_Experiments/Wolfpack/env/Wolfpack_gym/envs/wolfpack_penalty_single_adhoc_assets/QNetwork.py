import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl

class DQN(nn.Module):

    def __init__(self, h, w, hidden, lstm_seq_length, outputs, extended_feature_len=0, conv_kernel_sizes = [4,2],
                 pool_kernel_sizes=[3,2],  conv_strides=[1,1], pool_conv_strides=[1,1],
                 num_channels = 3, device="cpu", mode="full"):
        super(DQN, self).__init__()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_sizes = pool_kernel_sizes
        self.conv_strides = conv_strides
        self.pool_conv_strides = pool_conv_strides
        self.mode = mode
        self.device = device

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=self.conv_kernel_sizes[0],
                               stride=self.conv_strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool1 = nn.MaxPool2d(self.pool_kernel_sizes[0],
                                      stride=self.pool_conv_strides[0])
        self.conv2 = nn.Conv2d(16, 32, kernel_size= self.conv_kernel_sizes[1],
                               stride=self.conv_strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(self.pool_kernel_sizes[1],
                                      stride=self.pool_conv_strides[1])

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        def pooling_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def calculate_output_dim(inp):
            return pooling_size_out(conv2d_size_out(pooling_size_out(
            conv2d_size_out(inp, kernel_size=self.conv_kernel_sizes[0], stride = self.conv_strides[0]),
            kernel_size=self.pool_kernel_sizes[0], stride=self.pool_conv_strides[0]),
            kernel_size=self.conv_kernel_sizes[1], stride = self.conv_strides[1]),
            kernel_size=self.pool_kernel_sizes[1], stride=self.pool_conv_strides[1])

        convw = calculate_output_dim(w)
        convh = calculate_output_dim(h)


        self.lstm_input_dim = convw * convh * 32
        self.hidden_dim = hidden

        self.lstm = nn.LSTM(self.lstm_input_dim, hidden, batch_first=True)
        self.lstm_seq_length = lstm_seq_length
        self.head = nn.Linear(hidden+extended_feature_len, 7)
        #self.head2 = nn.Linear(20, outputs)



    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, extended_feature = None):
        original_inp_size = list(x.size())
        transformed_size = [original_inp_size[0]*original_inp_size[1]]
        transformed_size.extend(original_inp_size[2:])

        x = x.view(transformed_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)
        x = x.view(original_inp_size[0],original_inp_size[1],-1)

        hidden = (torch.zeros(1, original_inp_size[0], self.hidden_dim).to(self.device),
                         torch.zeros(1, original_inp_size[0], self.hidden_dim).to(self.device))

        x, hidden = self.lstm(x, hidden)

        input = x[:,-1,:]
        if not extended_feature is None :
            input = torch.cat((x[:,-1,:], extended_feature), dim=-1)
        #action_vals = F.relu(self.head(input))
        action_vals = self.head(input)
        return action_vals

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
                       in zip(g_repr,edge_nums)], dim=0)
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

class GraphLSTM(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, dim_out, unbatch_return_feats=True):
        super(GraphLSTM, self).__init__()
        self.lstm_edge = nn.LSTM(dim_in_edge, dim_out, batch_first=True)
        self.lstm_node = nn.LSTM(dim_in_node, dim_out, batch_first=True)
        self.lstm_u = nn.LSTM(dim_in_u, dim_out, batch_first=True)

        self.graph_msg = fn.copy_edge(edge='edge_feat', out='m')
        self.graph_reduce = fn.sum(msg='m', out='h')
        self.unbatch_return_feats = unbatch_return_feats

    def graph_message_func(self,edges):
        return {'m': edges.data['edge_feat']}

    def graph_reduce_func(self,nodes):
        msgs = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': msgs}

    def compute_edge_repr(self, graph, edges, g_repr):
        edge_nums = graph.batch_num_edges
        u = torch.cat([g[None, :].repeat(num_edge, 1) for g, num_edge
                       in zip(g_repr, edge_nums)], dim=0)
        inp = torch.cat([edges.data['edge_feat'], edges.src['node_feat'],
                         edges.dst['node_feat'], u], dim=-1)[:,None,:]
        hidden = (edges.data['hidden1'][None,:,:], edges.data['hidden2'][None,:,:])
        out, hidden = self.lstm_edge(inp, hidden)
        out_shape = out.shape
        out = out.view([out_shape[0], out_shape[2]])
        return {'edge_feat': F.relu(out), 'hidden1' : hidden[0][0], 'hidden2' : hidden[1][0]}

    def compute_node_repr(self, graph, nodes, g_repr):
        node_nums = graph.batch_num_nodes
        u = torch.cat([g[None, :].repeat(num_node, 1) for g, num_node
                       in zip(g_repr, node_nums)], dim=0)
        inp = torch.cat([nodes.data['node_feat'], nodes.data['h'], u], dim=-1)[:,None,:]
        hidden = (nodes.data['hidden1'][None,:,:], nodes.data['hidden2'][None,:,:])
        out, hidden = self.lstm_node(inp, hidden)
        out_shape = out.shape
        out = out.view([out_shape[0], out_shape[2]])
        return {'node_feat' : F.relu(out), 'hidden1' : hidden[0][0], 'hidden2' : hidden[1][0]}

    def compute_u_repr(self, n_comb, e_comb, g_repr, hidden):
        inp = torch.cat([n_comb, e_comb, g_repr], dim=-1)[:,None,:]
        out, hidden = self.lstm_u(inp, hidden)
        out_shape = out.shape
        out = out.view([out_shape[0], out_shape[2]])
        return F.relu(out), hidden


    def forward(self, graph, edge_feat, node_feat, g_repr, edge_hidden, node_hidden, graph_hidden):

        graph.edata['edge_feat'] = edge_feat
        graph.ndata['node_feat'] = node_feat
        graph.edata['hidden1'] = edge_hidden[0][0]
        graph.ndata['hidden1'] = node_hidden[0][0]
        graph.edata['hidden2'] = edge_hidden[1][0]
        graph.ndata['hidden2'] = node_hidden[1][0]

        node_trf_func = lambda x : self.compute_node_repr(nodes=x, graph=graph, g_repr=g_repr)
        edge_trf_func = lambda x: self.compute_edge_repr(edges=x, graph=graph, g_repr=g_repr)
        graph.apply_edges(edge_trf_func)
        graph.update_all(self.graph_message_func, self.graph_reduce_func, node_trf_func)

        e_comb = dgl.sum_edges(graph, 'edge_feat')
        n_comb = dgl.sum_nodes(graph, 'node_feat')

        u_out, u_hidden = self.compute_u_repr(n_comb, e_comb, g_repr, graph_hidden)

        e_feat = graph.edata['edge_feat']
        n_feat = graph.ndata['node_feat']

        h_e = (torch.unsqueeze(graph.edata['hidden1'],0),torch.unsqueeze(graph.edata['hidden2'],0))
        h_n =  (torch.unsqueeze(graph.ndata['hidden1'],0),torch.unsqueeze(graph.ndata['hidden2'],0))

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())
        for key in e_keys:
            graph.edata.pop(key)
        for key in n_keys:
            graph.ndata.pop(key)

        return e_feat, h_e, n_feat, h_n, u_out, u_hidden

class GraphOppoModel(nn.Module):
    def __init__(self,dim_in_node, dim_in_edge, dim_in_u, hidden_dim, hidden_dim2, dim_mid, dim_out,
                 added_mid_dims, act_dims):
        super(GraphOppoModel, self).__init__()
        self.GNBlock = RFMBlock(dim_mid + dim_in_node + dim_in_u,
                                2 * dim_in_node + dim_in_u + dim_in_edge,
                                2 * dim_mid + dim_in_u, hidden_dim,
                                dim_mid)
        self.GNBlock2 = RFMBlock(dim_out + 2 * dim_mid, 4 * dim_mid, 2 * dim_out + dim_mid, hidden_dim2, dim_out)

        self.head = nn.Linear(dim_out, added_mid_dims)
        self.head2 = nn.Linear(added_mid_dims, act_dims)

    def forward(self, graph, edge_feat, node_feat, u_obs):

        g_repr = u_obs
        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock.forward(graph, edge_feat, node_feat, g_repr)

        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock2.forward(graph,
                                                                               updated_e_feat, updated_n_feat,
                                                                               updated_u_feat)

        out = self.head2(F.relu(self.head(updated_n_feat)))
        return out

class RFMLSTMMiddle(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, hidden_dim, hidden_dim2, dim_lstm_out, dim_mid, dim_out,
                 fin_mid_dim, act_dims, with_added_u_feat=False, added_u_feat_dim=0):
        super(RFMLSTMMiddle, self).__init__()
        self.dim_lstm_out = dim_lstm_out
        self.with_added_u_feat = with_added_u_feat

        if not self.with_added_u_feat:
            self.GNBlock = RFMBlock(dim_mid+ dim_in_node + dim_in_u,
                                2 * dim_in_node + dim_in_u + dim_in_edge,
                               	2 * dim_mid + dim_in_u, hidden_dim,
                                dim_mid)
        else:
            self.GNBlock = RFMBlock(dim_mid + dim_in_node + dim_in_u + added_u_feat_dim,
                                    2 * dim_in_node + dim_in_u + dim_in_edge + added_u_feat_dim,
                                    2 * dim_mid + dim_in_u + added_u_feat_dim, hidden_dim,
                                    dim_mid)
        self.GraphLSTM = GraphLSTM(dim_lstm_out+2*dim_mid, 4*dim_mid, 2*dim_lstm_out + dim_mid, dim_lstm_out)
        self.GNBlock2 = RFMBlock(dim_out+2*dim_lstm_out, 4*dim_lstm_out, 2*dim_out + dim_lstm_out, hidden_dim2,
                                 dim_out)

        self.pre_q_net = nn.Linear(dim_out, fin_mid_dim)
        self.q_net = nn.Linear(fin_mid_dim, act_dims)


    def forward(self, graph, edge_feat, node_feat, u_obs, hidden_e, hidden_n, hidden_u, added_u_feat=None):

        g_repr = u_obs

        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock.forward(graph, edge_feat,
                                                                              node_feat, g_repr)
        updated_e_feat, e_hid, updated_n_feat, n_hid, updated_u_feat, u_hid = self.GraphLSTM.forward(graph,
                                                                              updated_e_feat, updated_n_feat,
                                                                              updated_u_feat, hidden_e,
                                                                              hidden_n, hidden_u)
        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock2.forward(graph, updated_e_feat,
                                                                               updated_n_feat, updated_u_feat)

        inp = updated_n_feat
        out = self.q_net(F.relu(self.pre_q_net(inp)))

        return out, e_hid, n_hid, u_hid

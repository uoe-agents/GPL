import torch
import torch.nn as nn
import torch.nn.functional as F

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
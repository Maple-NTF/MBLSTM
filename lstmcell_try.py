import torch
import torch.nn as nn

class InteractionModule(nn.Module):
    def __init__(self, units, num_iterations):
        super(InteractionModule, self).__init__()
        self.units = units
        self.num_iterations = num_iterations
        self.dense1 = nn.Linear(units, units)
        self.dense2 = nn.Linear(units, units)
        self.dense3 = nn.Linear(units, units)
        self.dense4 = nn.Linear(units, units)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, forward_hidden_state, backward_hidden_state):
        x0 = inputs
        # print(x0)
        hf = forward_hidden_state
        # print(hf)
        hb = backward_hidden_state
        # print(hb)

        for _ in range(self.num_iterations):
            # y1 = torch.cat([x0, hf], dim=-1)
            # print(y1.shape)
            # print(self.dense1)
            x1 = self.sigmoid(self.dense1(x0 + hf))
            # print(x1.shape)
            # print("************************")
            hb2 = self.sigmoid(self.dense2(hb + x1))
            # print(hb2.shape)
            # print("************************")
            hf2 = self.sigmoid(self.dense3(x1 + hf))
            # print(hf2.shape)
            # print("************************")
            x2 = self.sigmoid(self.dense4(hb2 + x1))
            # print(x2.shape)
            # print("￥￥￥￥￥￥￥￥￥￥￥￥￥￥")
            x0 = x2
            hf = hf2
            hb = hb2

            # print(x0)
            # print(hf)
            # print(hb)
        return x0, hf, hb

seq_len = 100
batch_size = 30

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_iterations, seq_len, batch_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.seq_len = seq_len
        self.batch_size = batch_size

        # 定义前向LSTM单元
        self.forward_lstmcell = nn.LSTMCell(input_size, hidden_size)

        # 定义后向LSTM单元
        self.backward_lstmcell = nn.LSTMCell(input_size, hidden_size)

        # 定义交互模块
        self.interaction_output_forward_h = torch.zeros(1)
        self.interaction_output_backward_h = torch.zeros(1)
        self.interaction_module = InteractionModule(hidden_size, num_iterations)

    def forward(self, inputs):
        # 初始化前向LSTM隐藏状态和细胞状态
        self.interaction_output_forward_h= torch.zeros(batch_size, self.hidden_size)
        forward_cell = torch.zeros(batch_size, self.hidden_size)

        # 初始化后向LSTM隐藏状态和细胞状态
        self.interaction_output_forward_h = torch.zeros(batch_size, self.hidden_size)
        backward_cell = torch.zeros(batch_size, self.hidden_size)

        # 存储交互模块的输出
        # interaction_outputs = []

        # 前向传播
        for t in range(seq_len):
            forward_seq_now = inputs[t]
            backward_seq_now = inputs[seq_len - t - 1]

            # forward_hidden, forward_cell = self.forward_lstmcell(forward_seq_now, (forward_hidden, forward_cell))
            # backward_hidden, backward_cell = self.backward_lstmcell(backward_seq_now, (backward_hidden, backward_cell))

            # 交互模块
            interaction_output, self.interaction_output_forward_h, self.interaction_output_backward_h = self.interaction_module(
                inputs, self.interaction_output_forward_h, self.interaction_output_backward_h)  # 获取了hf和hb
            interaction_output = (self.interaction_output_forward_h + self.interaction_output_backward_h) / 2
            #interaction_outputs.append(interaction_output)

            _, forward_cell = self.forward_lstmcell(forward_seq_now, (self.interaction_output_forward_h, forward_cell))
            _, backward_cell = self.backward_lstmcell(backward_seq_now, (self.interaction_output_backward_h, backward_cell))

        # 将交互模块的输出转换为张量
        return interaction_output

#这部分是将LSTM拆分为每个单元进行处理的第一段编写，里面仍然有许多内容是逻辑不通的，不是最终版本
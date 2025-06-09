import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, inputsize=512, hidensize=256, layer=1, outputsize=2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(inputsize, hidensize, layer)
        self.linear = nn.Linear(hidensize, outputsize)

    def forward(self, x):
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(x)
        output = self.linear(lstm_out)

        return output


class RnnModel(nn.Module):
    def __init__(self, inputsize=512, hidensize=256, layer=3, outputsize=2):
        super(RnnModel, self).__init__()

        self.rnn = nn.RNN(inputsize, hidensize, layer, nonlinearity='tanh')
        self.linear = nn.Linear(hidensize, outputsize)

    def forward(self, x):
        r_out, h_state = self.rnn(x)
        output = self.linear(r_out[:,-1,:])  # 将 RNN 层的输出 r_out 在最后一个时间步上的输出（隐藏状态）传递给线性层
        output = output.unsqueeze(1)

        return output

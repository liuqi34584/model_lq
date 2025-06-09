import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  #  将这个张量的梯度追踪关闭，因为位置嵌入是固定的，不需要在训练过程中更新

        position = torch.arange(0, max_len).float().unsqueeze(1)

        # 用来调整不同维度上位置编码的变化率,就是论文中位置嵌入的公式
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # pe: torch.Size([1, 1, 512])

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

        # 一种初始化权重方法，使得权重具有较好的初始值
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.position_embedding = LearnablePositionalEncoding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        # x = self.value_embedding(x)
        # x = self.position_embedding(x)

        return self.dropout(x)


class WaveletEmbeding(nn.Module):
    def __init__(self, c_in, d_model, scale_num, dropout=0.0):
        super(WaveletEmbeding, self).__init__()


        wavelet_y, wavelet_x = self.cmor_torch(-8,8, 1024, 2.0, 8.0, scale_num)  # -8,8,1024,2.0,2.0

        # 向模型注册持久性缓冲区（buffer）在训练过程中，缓冲区值是保持不变，不受梯度影响
        self.register_buffer('wavelet_y', torch.tensor(wavelet_y, dtype=torch.float32))
        self.register_buffer('wavelet_x', torch.tensor(wavelet_x, dtype=torch.int32))

        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x):  # torch.Size([8192, 1, 300])

        # # 单通道
        # out = x
        # for i, (s,e) in enumerate(self.wavelet_x):
        #     kernel = self.wavelet_y[s:e].unsqueeze(0).unsqueeze(0)
        #     conv1=F.conv1d(x, kernel, stride=1, padding = (kernel.size(-1) - 1) // 2 )
        #     conv1 = - torch.sqrt(torch.tensor(i+1)) * torch.diff(conv1, axis=-1)
        #     # conv1 = torch.diff(conv1, axis=-1)
        #     conv1 = torch.abs(conv1)

        #     conv1 = torch.cat([conv1, conv1[:,:,-int(x.size(-1)-conv1.size(-1)):]], dim=-1)  # reflect 填充到 x.size(-1) 大小
        #     out = torch.cat([out, conv1], dim=1)

        # out = self.tokenConv(out.permute(0, 2, 1)).transpose(1, 2) # [8192, 1, 300] --> [8192, 1, 512] 

        # 单多通道兼容
        out = x
        for split_index in torch.arange(x.size(1)):
            for i, (s,e) in enumerate(self.wavelet_x):
                x1 = x[:,split_index:split_index+1,:]
                kernel = self.wavelet_y[s:e].unsqueeze(0).unsqueeze(0)
                conv1=F.conv1d(x1, kernel, stride=1, padding = (kernel.size(-1) - 1) // 2 )
                # conv1 = - torch.sqrt(torch.tensor(i+1)) * torch.diff(conv1, axis=-1)
                conv1 = torch.diff(conv1, axis=-1)
                conv1 = torch.abs(conv1)

                conv1 = torch.cat([conv1, conv1[:,:,-int(x.size(-1)-conv1.size(-1)):]], dim=-1)  # reflect 填充到 x.size(-1) 大小
                out = torch.cat([out, conv1], dim=1)


        # import matplotlib.pyplot as plt

        # plt.subplot(5,1,1)
        # plt.plot(out[0,0,:].cpu().detach().numpy())

        # plt.subplot(5,1,2)
        # plt.plot(out[0,1,:].cpu().detach().numpy())

        # plt.subplot(5,1,3)
        # plt.plot(out[0,2,:].cpu().detach().numpy())

        # plt.subplot(5,1,4)
        # plt.plot(out[0,3,:].cpu().detach().numpy())

        # plt.subplot(5,1,5)
        # plt.plot(out[0,4,:].cpu().detach().numpy())

        # plt.show()


        # out = self.tokenConv(out.permute(0, 2, 1)).transpose(1, 2) + self.position_embedding(out)  # [8192, 1, 300] --> [8192, 1, 512] 
        out = self.tokenConv(out.permute(0, 2, 1)).transpose(1, 2)  # [8192, 1, 300] --> [8192, 1, 512] 

        return out

    def cmor_torch(self, LB, UB, N, Fb, Fc, scale_num):

        wavelet_y, wavelet_x = [], []
        x = np.linspace(LB, UB, N)
        wavelet_ori = ((np.pi*Fb)**(-0.5))*np.exp(2j*np.pi*Fc*x)*np.exp(-(x**2)/Fb)
        for i, scale in enumerate(np.arange(1, scale_num)):
            step = x[1] - x[0]
            j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)  # 坐标轴伸，步距也伸，才是尺度变换
            j = j.astype(int)
            if j[-1] >= wavelet_ori.size:
                j = np.extract(j < wavelet_ori.size, j)  
                # 从数组 j 中提取出所有小于 int_psi 大小（长度）的元素
                # j 索引如何被放缩， 5000/ 5 = 【1  5 10 15 20 ...】
                # j 索引如何被放缩，10000/10 = 【1 10 20 30 40 ...】

            s = len(wavelet_y)
            wavelet_y.extend(np.real(wavelet_ori[j]*((scale)**(-0.5))))  # 在这里选项了实数小波
            e =  len(wavelet_y)
            wavelet_x.append([s,e])

            # print(len(wavelet_y))
            # print(wavelet_x)

            # import matplotlib.pyplot as plt
            # plt.plot(np.real(wavelet_ori[j]*((scale)**(-0.5))))
            # plt.show()

        return wavelet_y, wavelet_x


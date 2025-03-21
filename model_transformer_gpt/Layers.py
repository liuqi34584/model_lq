''' Define the Layers '''
import torch.nn as nn
import torch
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward
import math
from math import sqrt
import torch.nn.functional as F
import torch
from .Modules import ScaledDotProductAttention


class My_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, sigma, u, d_model, d_inner, n_head, d_k, d_v, dropout):
        super(My_EncoderLayer, self).__init__()

        self.guass_length = n_head # 64
        self.guass_hide = d_k # 8

        # 建立完毕，必须要注册缓冲区 buffer，不然一开始数据不在 GPU 上
        self.guass_2, self.guass_3 = self.gauss(length=n_head, sigma=sigma , u=u, hide=d_k)
        self.register_buffer('guass_2d', self.guass_2)  # torch.Size([guass_length, guass_length])
        self.register_buffer('gauss_3d', self.guass_3)  # torch.Size([guass_length, guass_length, guass_hide])

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.WQ = nn.Linear(d_k, 1, bias=True)  # 此处偏置打开，其性能变好
        self.WK = nn.Linear(d_k, 1, bias=True)  # 此处偏置打开，其性能变好
        self.WV = nn.Linear(d_k, 1, bias=True)  # 此处偏置打开，其性能变好

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.conv3d = nn.Conv3d(in_channels=4, out_channels=4, groups=1, kernel_size=(1, 96, 8), bias=False)

    def forward(self, enc_input):  # torch.Size([512, 5, 512])

        B,C,L,H = enc_input.size(0), enc_input.size(1), self.guass_length, self.guass_hide # 512, 5, 64, 8

        heads = enc_input.unfold(-1, H, H)  # 切片长度为 H, 步长为 H, torch.Size([B, C, nhead, d_k])

        # gauss
        # key_mask =self.gauss_3d.expand(B, C, L, L, H)   # torch.Size([B, C, nhead, nhead, d_k])
        # key_data = heads.unsqueeze(2).expand(-1, -1, self.guass_length, -1, -1)  # torch.Size([B, C, nhead, nhead, d_k])
        # key_data = key_mask*key_data  # torch.Size([B, C, nhead, nhead, d_k])

        # key = self.w_ks(key_data.permute(0,1,2,4,3)).squeeze()  # torch.Size([B, C, nhead, d_k])

######################
        # print("key_data", key_data.shape)
        # key = self.conv3d(key_data).squeeze()
        # print("key", "heads", key.shape, heads.shape)

######################
        # key = key.unsqueeze(-1).expand_as(heads)
        # result = torch.mul(key, heads)
        # result += heads 
        # enc_output = self.layer_norm(result.contiguous().view(B, C, -1))
        # enc_output = self.pos_ffn(enc_output)

######################

        query_data = self.WQ(heads).squeeze()  # torch.Size([B, C, nhead])
        vector_data = self.WV(heads).squeeze()  # torch.Size([B, C, nhead])

        key_mask =self.guass_2d.expand(B, C, L, L)  # torch.Size([B, C, nhead, nhead])
        key_data = self.WK(heads).squeeze()  # torch.Size([B, C, nhead])

        # 横着扩展，竖着扩展，然后相乘，得到外积特征图
        a = key_data.unsqueeze(-1).expand(-1, -1, -1, self.guass_length)  # torch.Size([B, C, nhead, nhead])
        b = query_data.unsqueeze(-1).expand(-1, -1, -1, self.guass_length)  # torch.Size([B, C, nhead, nhead])
        feature_map = torch.mul(b, a.permute(0,1,3,2))



        # 这一块是对特征图进行归一化，先铺平，再归一化，再返回到二维特征图
        x_normalized = F.normalize(feature_map.view(B, C, -1), p=2, dim=2)
        feature_map = x_normalized.view(B, C, L, L)

        # feature_map_show = x_normalized.view(B, C, L, L)
        # import pandas as pd
        # df = pd.DataFrame(feature_map_show[0,0,:,:].cpu().detach().numpy())
        # df.to_excel('./output/transformer_encoder_MITBIHAF/feature_map_show.xlsx', index=False)

        feature_map = torch.mul(key_mask, feature_map)

        # score_show = torch.mul(key_mask, feature_map)
        # import pandas as pd
        # df = pd.DataFrame(score_show[0,0,:,:].cpu().detach().numpy())
        # df.to_excel('./output/transformer_encoder_MITBIHAF/score_show.xlsx', index=False)


        feature_map = torch.sum(feature_map, dim=3)  # [B, C, nhead]
        attn =F.softmax(feature_map, dim=2)  # [B, C, nhead]

        result = torch.mul(attn, vector_data)

        result = result.unsqueeze(-1).expand_as(heads)
        result = result + heads.clone()
        enc_output = self.layer_norm(result.contiguous().view(B, C, -1))
        enc_output = self.pos_ffn(enc_output)


        # # 展示原特征和注意力之后的特征变化
        # import matplotlib.pyplot as plt
        # plt.subplot(1,3,1)
        # plt.imshow(feature_map_show[0,0,:,:].cpu().detach().numpy(), cmap='plasma')
        # plt.subplot(1,3,2)
        # plt.imshow(key_mask[0,0,:,:].cpu().detach().numpy(), cmap='plasma')
        # plt.subplot(1,3,3)
        # plt.imshow(score_show[0,0,:,:].cpu().detach().numpy(), cmap='plasma')
        # plt.show()

        return enc_output

    def gauss(self, length, sigma, u, hide):

        x = torch.tensor([i for i in torch.arange(-length, length)])
        guass = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-(x-u) ** 2 / 2 / (sigma ** 2))
        guass = torch.div(guass, torch.max(guass))  # 将高斯分布 guass 进行归一化处理，使得分布在 [0, 1] 范围内

        # # 将其填为全为 1
        # x = torch.arange(-length, length + 1)
        # guass = torch.ones_like(x, dtype=torch.float32)

        # # 展示高斯函数分布
        # import matplotlib.pyplot as plt
        # plt.plot(guass.cpu().detach().numpy())
        # plt.show()

        window = []
        for i in torch.arange(length):
            step = guass[i:i+length]
            window.append(step)  # 登记每行 高斯分布

        guass_2d = torch.stack(window[::-1], dim=0)
        # guass_2d = torch.bernoulli(torch.stack(window[::-1], dim=0))  # 伯努利分布 0-1 独立

        # 展示高斯分布图
        # import matplotlib.pyplot as plt
        # plt.imshow(guass_2d.cpu().detach().numpy(), cmap='plasma')
        # plt.show()

        # 保存高斯分布
        # import pandas as pd
        # df = pd.DataFrame(guass_2d.numpy())
        # df.to_excel('./output/transformer_encoder_MITBIHAF/gauss2d.xlsx', index=False)

        guass_3d = torch.bernoulli(guass_2d.unsqueeze(2).expand(length, length, hide))  # 伯努利随机矩阵

        # gauss

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        # # 获取非零元素的坐标
        # indices = guass_3d.cpu().detach().numpy().nonzero()
        # # 创建一个 3D 图
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # 绘制散点图
        # ax.scatter(*indices, marker='*')
        
        # # 设置坐标轴标签
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # # 显示图形
        # plt.show()

        return guass_2d, guass_3d 

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        # q0, k0, v0 = enc_input[:,0,:].unsqueeze(1), enc_input[:,0,:].unsqueeze(1), enc_input[:,0,:].unsqueeze(1)
        # q1, k1, v1 = enc_input[:,1,:].unsqueeze(1), enc_input[:,1,:].unsqueeze(1), enc_input[:,1,:].unsqueeze(1)
        # q2, k2, v2 = enc_input[:,2,:].unsqueeze(1), enc_input[:,2,:].unsqueeze(1), enc_input[:,2,:].unsqueeze(1)
        # q3, k3, v3 = enc_input[:,3,:].unsqueeze(1), enc_input[:,3,:].unsqueeze(1), enc_input[:,3,:].unsqueeze(1)
        # q4, k4, v4 = enc_input[:,4,:].unsqueeze(1), enc_input[:,4,:].unsqueeze(1), enc_input[:,4,:].unsqueeze(1)
        # # enc_input torch.Size([512, 5, 180]) q0 torch.Size([512, 1, 180])

        # enc_output0, enc_slf_attn0 = self.slf_attn(q0, k0, v0, mask=slf_attn_mask)
        # enc_output1, enc_slf_attn1 = self.slf_attn(q1, k1, v1, mask=slf_attn_mask)
        # enc_output2, enc_slf_attn2 = self.slf_attn(q2, k2, v2, mask=slf_attn_mask)
        # enc_output3, enc_slf_attn3 = self.slf_attn(q3, k3, v3, mask=slf_attn_mask)
        # enc_output4, enc_slf_attn4 = self.slf_attn(q4, k4, v4, mask=slf_attn_mask)

        # enc_output = torch.cat([enc_output0, enc_output1, enc_output2, enc_output3, enc_output4], dim=1)
        # enc_slf_attn = torch.cat([enc_slf_attn0, enc_slf_attn1, enc_slf_attn2, enc_slf_attn3, enc_slf_attn4], dim=1)

        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


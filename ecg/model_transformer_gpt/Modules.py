import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        # print(q.shape, k.shape, v.shape)  # torch.Size([8192, 8, 1, 64]) torch.Size([8192, 8, 1, 64]) torch.Size([8192, 8, 1, 64])
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # print("attn:", attn.shape)  # attn torch.Size([8192, 8, 1, 1])

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        # print("softmax attn:", attn.shape)  # softmax attn: torch.Size([8192, 8, 1, 1])
        output = torch.matmul(attn, v)
        # print("output:", output.shape)  # output: torch.Size([8192, 8, 1, 64])

        return output, attn

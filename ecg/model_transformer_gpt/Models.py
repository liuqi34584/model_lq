''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer, My_EncoderLayer
from .embed import DataEmbedding,WaveletEmbeding
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import torch.nn.functional as F
import math
from math import sqrt

class MY_Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, sigma, u, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.my_attn = nn.ModuleList([
            My_EncoderLayer(sigma, u, d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_in):

        enc_output = enc_in
    
        for enc_layer in self.my_attn:
            enc_output = enc_layer(enc_output)

        return enc_output

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_in):

        enc_output = enc_in

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)

        return enc_output

class Transformer(nn.Module):
    def __init__(self, c_in=768, c_out=2, d_model=768, d_inner=2048, n_layers=1, n_head=8, d_k=64, d_v=64, dropout=0):
        super().__init__()
        self.d_model = d_model

        # # *********************************************************** transformer ******************************************************************
        # self.embedding = DataEmbedding(c_in, d_model, dropout)
        # self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        # self.linear_reshape = nn.Linear(d_model, c_out, bias=False)

        # # *********************************************************** my_WaveletEmbeding ***********************************************************
        # scale_num = 5
        # self.embedding = WaveletEmbeding(c_in, d_model, scale_num, dropout)
        # self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        # self.linear_reshape = nn.Linear(d_model*scale_num, c_out, bias=False)

        # # ***************************************************************** gpt2.0 *******************************************************************
        # # GPT模型需要输入长度768，因此将c_in设置为768 d_model设置为768
        # self.embedding = DataEmbedding(c_in, d_model, dropout)
        # config = GPT2Config.from_json_file("other_file\gpt2\config.json")
        # config.n_embd = d_model
        # config.n_head = n_head
        # config.n_layer = n_layers

        # self.gpt2 = GPT2Model.from_pretrained('other_file\gpt2', config=config)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()
        # self.gpt2.to(device=device)

        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     # if 'ln' in name:  # 只将 layer_normal 的梯度打开
        #     #     param.requires_grad = True
        #     # else:
        #     #     param.requires_grad = False
        #     param.requires_grad = True

        # self.act = F.gelu
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.linear_reshape = nn.Linear(d_model, c_out)

        # # *********************************************************** my_WaveletEmbeding_gpt2.0 ******************************************************************
        # # GPT模型需要输入长度768，因此将c_in设置为768 d_model设置为768
        # scale_num = 5
        # self.embedding = WaveletEmbeding(c_in, d_model, scale_num, dropout)
        # config = GPT2Config.from_json_file("C:\mycode\ECG_test\gpt2_test\gpt2\config.json")
        # config.n_embd = d_model
        # config.n_head = n_head
        # config.n_layer = n_layers

        # self.gpt2 = GPT2Model.from_pretrained('C:\mycode\ECG_test\gpt2_test\gpt2', config=config)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()
        # self.gpt2.to(device=device)

        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     # if 'ln' in name:  # 只将 layer_normal 的梯度打开
        #     #     param.requires_grad = True
        #     # else:
        #     #     param.requires_grad = False
        #     param.requires_grad = True

        # self.act = F.gelu
        # self.layer_norm = nn.LayerNorm(d_model*scale_num, eps=1e-6)
        # self.linear_reshape = nn.Linear(d_model*scale_num, c_out)

        # *********************************************************** my_WaveletEmbeding + my_guass ******************************************************************
        # # 自开发模块，建议将c_in设置为512 d_model设置为512，scale_num 设置为 5
        # scale_num = 5
        # self.embedding = WaveletEmbeding(c_in, d_model, scale_num, dropout)
        # # self.embedding = DataEmbedding(c_in, d_model, dropout)
        # self.encoder = MY_Encoder(sigma=32 , u=0, d_model=512, d_inner=2048, n_layers=1, n_head=64, d_k=8, d_v=8, dropout=0)  # 小论文 n_layers=6
        # # self.linear_reshape = nn.Linear(d_model*scale_num*5, c_out, bias=False)
        # self.linear_reshape = nn.Linear(d_model*scale_num, c_out, bias=False)


        # *********************************************************** my_WaveletEmbeding + my_guass + gpt2.0 ******************************************************************
        # GPT模型需要输入长度768，因此将c_in设置为768 d_model设置为768，scale_num 设置为 5
        scale_num = 5
        self.embedding = WaveletEmbeding(c_in, d_model, scale_num, dropout)
        self.encoder = MY_Encoder(sigma=32 , u=0, d_model=768, d_inner=2048, n_layers=1, n_head=96, d_k=8, d_v=8, dropout=0)
        config = GPT2Config.from_json_file("C:\mycode\ECG_test\gpt2_test\gpt2\config.json")
        config.n_embd = d_model
        config.n_head = n_head
        config.n_layer = n_layers

        self.gpt2 = GPT2Model.from_pretrained('C:\mycode\ECG_test\gpt2_test\gpt2', config=config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()
        self.gpt2.to(device=device)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            # if 'ln' in name:  # 只将 layer_normal 的梯度打开
            #     param.requires_grad = True
            # else:
            #     param.requires_grad = False
            param.requires_grad = True

        self.act = F.gelu
        self.layer_norm = nn.LayerNorm(d_model*scale_num, eps=1e-6)
        self.linear_reshape = nn.Linear(d_model*scale_num, c_out, bias=False)









        # 这一步很重要，让损失下降变得稳定了
        # Xavier初始化是一种常用的权重初始化方法，旨在使得每一层的输出方差保持一致，有助于避免梯度消失或梯度爆炸问题，并促进训练的稳定性和收敛性。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 


    def forward(self, input_signal):  # torch.Size([512, 5, 300])

        # # *********************************************************** transformer ******************************************************************
        # enc_output = self.embedding(input_signal)  # torch.Size([512, 1, 512])
        # enc_output = self.encoder(enc_output)
        # enc_output = enc_output.reshape(enc_output.size(0), 1, -1)
        # output = self.linear_reshape(enc_output)
        # output *= self.d_model ** -0.5

        # # *********************************************************** my_WaveletEmbeding ******************************************************************
        # enc_output = self.embedding(input_signal)  # torch.Size([512, 1, 512])
        # enc_output = self.encoder(enc_output)  # torch.Size([512, scale_num, 512])
        # enc_output = enc_output.reshape(enc_output.size(0), 1, -1)
        # output = self.linear_reshape(enc_output)
        # output = torch.softmax(output, dim=2)  # 最后一维概率总和为1

        # # *********************************************************** gpt2.0 ******************************************************************
        # enc_output = self.embedding(input_signal)
        # gpt_outputs = self.gpt2(inputs_embeds=enc_output, output_attentions=True, output_hidden_states=True).last_hidden_state
        # gpt_outputs = self.act(gpt_outputs)
        # gpt_outputs = gpt_outputs.reshape(gpt_outputs.size(0), 1, -1)
        # gpt_outputs = self.layer_norm(gpt_outputs)
        # output = self.linear_reshape(gpt_outputs)
        # output = torch.softmax(output, dim=2)  # 最后一维概率总和为1

        # # *********************************************************** my_WaveletEmbeding_gpt2.0 ******************************************************************
        # enc_output = self.embedding(input_signal)
        # gpt_outputs = self.gpt2(inputs_embeds=enc_output, output_attentions=True, output_hidden_states=True).last_hidden_state
        # gpt_outputs = self.act(gpt_outputs)
        # gpt_outputs = gpt_outputs.reshape(gpt_outputs.size(0), 1, -1)
        # gpt_outputs = self.layer_norm(gpt_outputs)
        # output = self.linear_reshape(gpt_outputs)
        # output = torch.softmax(output, dim=2)  # 最后一维概率总和为1

        # # *********************************************************** my_WaveletEmbeding + my_guass ******************************************************************
        # enc_outputs = self.embedding(input_signal)
        # enc_outputs = self.encoder(enc_outputs)
        # output = enc_outputs.reshape(enc_outputs.size(0), 1, -1)
        # output = self.linear_reshape(output)
        # output = torch.softmax(output, dim=2)  # 最后一维概率总和为1

        # *********************************************************** my_WaveletEmbeding + my_guass + gpt2.0 ******************************************************************
        enc_output = self.embedding(input_signal)
        enc_output = self.encoder(enc_output)  # torch.Size([512, 5, 768])
        gpt_outputs = self.gpt2(inputs_embeds=enc_output, output_attentions=True, output_hidden_states=True).last_hidden_state
        gpt_outputs = self.act(gpt_outputs)
        gpt_outputs = gpt_outputs.reshape(gpt_outputs.size(0), 1, -1)
        gpt_outputs = self.layer_norm(gpt_outputs)
        output = self.linear_reshape(gpt_outputs)
        output = torch.softmax(output, dim=2)  # 最后一维概率总和为1


        return output

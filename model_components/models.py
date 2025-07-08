import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import copy

def get_data(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    
    b_left = data[0,:,:] # x = 0
    b_right = data[-1,:,:] # x = 1
    b_upper = data[:,-1,:] # t = 1
    b_lower = data[:,0,:] # t = 0
    res = data.reshape(-1,2)

    return res, b_left, b_right, b_upper, b_lower


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn_param=1
        for s in list(p.size()):
            nn_param = nn_param*s
        pp += nn_param
    return pp


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs): 
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)
    

class DecoderOnlyPINNsformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, in_features=2):
        super(DecoderOnlyPINNsformer, self).__init__()
        
        self.in_features = in_features
        self.linear_emb = nn.Linear(self.in_features, d_model)
        # Simple positional embedding as another linear layer.
        self.pos_emb = nn.Linear(self.in_features, d_model)

        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, *inputs):
        src = torch.cat(inputs, dim=-1)
        # sanity check
        if src.size(-1) != self.in_features:
            raise ValueError(
                f"Expected concatenated inputs to have last-dim={self.in_features}, "
                f"but got {src.size(-1)}"
            )
        token_emb = self.linear_emb(src)
        pos_emb = self.pos_emb(src)
        src = token_emb + pos_emb

        d_output = self.decoder(src, src)  # decoder attends to input only
        output = self.linear_out(d_output)
        return output
    
# Ripped from PINNsformer github
class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINNs, self).__init__()

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, *inputs):
        src = torch.cat(inputs, dim=-1)
        return self.linear(src)
    

class FourierPINNsformer2D(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, d_in, mapping_size, x_range = (1.0, 8.0), y_range = (-2.0, 2.0), t_range = (0.0, 19.9)):
        super(FourierPINNsformer2D, self).__init__()

        self.in_features = d_in
        self.spatial_features = d_in - 1
        self.time_features = 1
        self.fourier_emb = nn.Linear(2 * mapping_size, d_model)
        self.pos_emb = nn.Linear(self.in_features, d_model)
        

        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        )
        
        # Add a random matrix B that is (in_features x mapping_size), and cannot be learned.
        self.register_buffer('B', torch.randn(self.in_features, mapping_size))
        
        # Define the min and max values for normalization
        # These values are based on the dataset used in the paper for the 2D Navier-Stokes equations.
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.t_min, self.t_max = t_range
        

    def forward(self, x, y, t):
        # Base normalization
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        spacetime = torch.cat([x_norm, y_norm, t_norm], dim=-1)
        src = 2 * torch.pi * spacetime @ self.B  # (batch_size, seq_len, mapping_size)
        f = torch.cat([src.sin(), src.cos()], dim=-1)  # (batch, seq_len, 2*mapping_size)
        emb_f = self.fourier_emb(f)  # (batch_size, seq_len, d_model)
        emb_p = self.pos_emb(spacetime)
        out = emb_f + emb_p

        d_output = self.decoder(out, out)  # decoder attends to input only
        output = self.linear_out(d_output)
        return output
    
    
    
class FourierPINNsformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, d_in, mapping_size, x_range = (0.0, 2*torch.pi), t_range = (0.0, 1.0)):
        super(FourierPINNsformer, self).__init__()

        self.in_features = d_in
        self.spatial_features = d_in - 1
        self.time_features = 1
        self.fourier_emb = nn.Linear(2 * mapping_size, d_model)
        self.pos_emb = nn.Linear(self.in_features, d_model)        

        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        )
        
        # Add a random matrix B that is (in_features x mapping_size), and cannot be learned.
        self.register_buffer('B', torch.randn(self.in_features, mapping_size))
        
        self.x_min, self.x_max = x_range
        self.t_min, self.t_max = t_range
        

    def forward(self, x, t):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
    
        spacetime = torch.cat([x_norm, t_norm], dim=-1)
        src = 2 * torch.pi * spacetime @ self.B  # (batch_size, seq_len, mapping_size)
        f = torch.cat([src.sin(), src.cos()], dim=-1)  # (batch, seq_len, 2*mapping_size)
        emb_f = self.fourier_emb(f)  # (batch_size, seq_len, d_model)
        emb_p = self.pos_emb(spacetime)
        out = emb_f + emb_p

        d_output = self.decoder(out, out)  # decoder attends to input only
        output = self.linear_out(d_output)
        return output
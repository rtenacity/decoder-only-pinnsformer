import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import scipy.io
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
    
    
class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size=32, init_scale=0.1):
        super(FourierFeatures, self).__init__()
        # Instead of one scalar, use a vector of scales (one per frequency band)
        self.scale = nn.Parameter(torch.ones(mapping_size) * init_scale, requires_grad=True)
        self.B = nn.Parameter(torch.randn(in_features, mapping_size), requires_grad=True)

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ (self.B * self.scale)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class EnhancedEmbedding(nn.Module):
    def __init__(self, in_features, d_model, mapping_size=32, init_scale=0.1):
        super(EnhancedEmbedding, self).__init__()
        self.fourier = FourierFeatures(in_features, mapping_size, init_scale)
        # Adjust the linear layer to account for the increased dimensionality (2*mapping_size)
        self.linear = nn.Linear(2 * mapping_size, d_model)
        self.pos_emb = nn.Linear(in_features, d_model)

    def forward(self, x):
        fourier_features = self.fourier(x)
        token_emb = self.linear(fourier_features)
        pos_emb = self.pos_emb(x)
        return token_emb + pos_emb


class DecoderOnlyPINNsformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads):
        super(DecoderOnlyPINNsformer, self).__init__()

        self.linear_emb = nn.Linear(2, d_model)
        # Simple positional embedding as another linear layer.
        self.pos_emb = nn.Linear(2, d_model)

        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        token_emb = self.linear_emb(src)
        pos_emb = self.pos_emb(src)
        src = token_emb + pos_emb

        d_output = self.decoder(src, src)  # decoder attends to input only
        output = self.linear_out(d_output)
        return output

class DecoderOnlyPINNsformerFourier(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, init_scale=0.1, mapping_size=32):
        super(DecoderOnlyPINNsformerFourier, self).__init__()

        # Use the EnhancedEmbedding module which combines Fourier features and a learnable positional embedding.
        self.embedding = EnhancedEmbedding(in_features=3, d_model=d_model, mapping_size=mapping_size, init_scale=init_scale)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x, y, t):
        src = torch.cat((x, y, t), dim=-1)
        src = self.embedding(src)
        d_output = self.decoder(src, src)
        output = self.linear_out(d_output)
        return output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = scipy.io.loadmat('./cylinder_nektar_wake.mat')
U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1

idx = np.random.choice(N*T,2500, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]

x_train = np.expand_dims(np.tile(x_train[:], (5)) ,-1)
y_train = np.expand_dims(np.tile(y_train[:], (5)) ,-1)
t_train = make_time_sequence(t_train, num_step=5, step=1e-2)

x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
u_train = torch.tensor(u_train, dtype=torch.float32, requires_grad=True).to(device)
v_train = torch.tensor(v_train, dtype=torch.float32, requires_grad=True).to(device)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
snap = np.array([100])
x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = U_star[:,0,snap]
v_star = U_star[:,1,snap]
p_star = P_star[:,snap]

x_star = np.expand_dims(np.tile(x_star[:], (5)) ,-1)
y_star = np.expand_dims(np.tile(y_star[:], (5)) ,-1)
t_star = make_time_sequence(t_star, num_step=5, step=1e-2)

x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(device)
y_star = torch.tensor(y_star, dtype=torch.float32, requires_grad=True).to(device)
t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True).to(device)

# Note: d_model (here 32) should be even for the Fourier features mapping.
model = DecoderOnlyPINNsformerFourier(d_out=2, d_hidden=512, d_model=32, N=1, heads=2, init_scale = 0.05).to(device)

model.apply(init_weights)
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
# optim = Adam(model.parameters(), lr = 1e-4)

print(model)
print(get_n_params(model))

n_params = get_n_params(model)
loss_track = []

progress_bar = tqdm(range(1000))

error_u, error_v, error_p = None, None, None
for i in progress_bar:
    
    if i % 5 == 0:
        psi_and_p = model(x_star, y_star, t_star)
        psi = psi_and_p[:,:,0:1]
        p_pred = psi_and_p[:,:,1:2]

        u_pred = torch.autograd.grad(psi, x_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v_pred = - torch.autograd.grad(psi, y_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

        u_pred = u_pred.cpu().detach().numpy()[:,0]
        v_pred = v_pred.cpu().detach().numpy()[:,0]
        p_pred = p_pred.cpu().detach().numpy()[:,0]


        error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
        error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
        error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
        
    if i % 20 == 0 and i > 0: 
        # save model weights
        torch.save(model.state_dict(), f'./ns_pinnsformer_{i}.pt')
        # save loss
        np.save(f'./ns_loss_pinnsformer_{i}.npy', loss_track)
    
    def closure():
        
        psi_and_p = model(x_train, y_train, t_train)
        psi = psi_and_p[:,:,0:1]
        p = psi_and_p[:,:,1:2]

        u = torch.autograd.grad(psi, y_train, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v = - torch.autograd.grad(psi, x_train, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

        u_t = torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

        v_t = torch.autograd.grad(v, t_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

        p_x = torch.autograd.grad(p, x_train, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y_train, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        f_u = u_t + (u*u_x + v*u_y) + p_x - 0.01*(u_xx + u_yy) 
        f_v = v_t + (u*v_x + v*v_y) + p_y - 0.01*(v_xx + v_yy)

        loss = torch.mean((u[:,0] - u_train)**2) + torch.mean((v[:,0] - v_train)**2) + torch.mean(f_u**2) + torch.mean(f_v**2)

        loss_track.append(loss.item())

        optim.zero_grad()
        loss.backward()
        return loss
    


    optim.step(closure)
    
    last_loss = loss_track[-1]
    progress_bar.set_postfix({
        "res": f"{last_loss:.6f}",
        "p": f"{error_p:.6f}",

    })
    
np.save('./ns_loss_pinnsformer.npy', loss_track)
torch.save(model.state_dict(), './ns_pinnsformer.pt')

print(loss_track[-1])

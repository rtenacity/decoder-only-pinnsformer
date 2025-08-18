import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import scipy.io
import copy
from model_components.models import *
from model_components.util import *
import optuna

class DecoderOnlyPINNsformerSpaceTime(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, d_in, mapping_size):
        super(DecoderOnlyPINNsformerSpaceTime, self).__init__()

        self.in_features = d_in
        self.spatial_features = d_in - 1
        self.time_features = 1
        self.fourier_emb = nn.Linear(2 * mapping_size, d_model)
        self.time_emb = nn.Linear(self.in_features, d_model)
        # self.test_emb = nn.Linear(self.in_features, d_model)  # For testing purposes, if needed
        

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
        
        self.x_min = 1.0
        self.x_max = 8.0
        self.y_min = -2.0
        self.y_max = 2.0
        self.t_min = 0.0
        self.t_max = 19.9
        

    def forward(self, x, y, t):
        # Base normalization
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        spacetime = torch.cat([x_norm, y_norm, t_norm], dim=-1)
        src = 2 * torch.pi * spacetime @ self.B  # (batch_size, seq_len, mapping_size)
        f = torch.cat([src.sin(), src.cos()], dim=-1)  # (batch, seq_len, 2*mapping_size)
        token_emb = self.fourier_emb(f)  # (batch_size, seq_len
        # token_emb = self.test_emb(spacetime)  # For testing purposes, if needed
        pos_emb = self.time_emb(spacetime)

        out = token_emb + pos_emb

        d_output = self.decoder(out, out)  # decoder attends to input only
        output = self.linear_out(d_output)
        return output
    
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
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


x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
t_min, t_max = t.min(), t.max()

print(f"x ∈ [{x_min}, {x_max}]")
print(f"y ∈ [{y_min}, {y_max}]")
print(f"t ∈ [{t_min}, {t_max}]")

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

smallest_rl1 = 1e10  # Initialize a large value to track the smallest L1 error


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def objective(trial): 
    global x_train, y_train, t_train, u_train, v_train, x_star, y_star, t_star, u_star, v_star, p_star, device, smallest_rl1
    set_seed(0)
    
    # sample hyperparameters
    d_hidden = trial.suggest_categorical('d_hidden', [128, 256, 512, 768])
    d_model = trial.suggest_categorical('d_model', [16, 32, 64, 128])
    mapping_size = trial.suggest_int('mapping_size', 16, 128, step=16)



    # Note: d_model (here 32) should be even for the Fourier features mapping.
    model = DecoderOnlyPINNsformerSpaceTime(d_out=2, d_hidden=d_hidden, d_model=d_model, N=1, heads=2, d_in=3, mapping_size=mapping_size).to(device)

    model.apply(init_weights)
    optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')



    for i in range(1000):
        
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


            optim.zero_grad()
            loss.backward()
            return loss
    

        optim.step(closure)
        

    def find_optimal_pressure_correction(p_true, p_pred):
        p_true_flat = p_true.flatten()
        p_pred_flat = p_pred.flatten()
        
        C = np.mean(p_true_flat - p_pred_flat)
        return C


    psi_and_p = model(x_star, y_star, t_star)
    psi = psi_and_p[:,:,0:1]
    p_pred = psi_and_p[:,:,1:2]


    u_pred = torch.autograd.grad(psi, x_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
    v_pred = - torch.autograd.grad(psi, y_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

    u_pred = u_pred.cpu().detach().numpy()[:,0]
    v_pred = v_pred.cpu().detach().numpy()[:,0]
    p_pred = p_pred.cpu().detach().numpy()[:,0]



    C_opt = find_optimal_pressure_correction(p_star, p_pred)
    p_pred = p_pred + C_opt


    rl1 = np.linalg.norm(p_star-p_pred,1)/np.linalg.norm(p_star,1)
    
    if rl1 < smallest_rl1:
        smallest_rl1 = rl1
        torch.save(model.state_dict(), f'saves/ns-spformer-trial{trial.number}.pth')
    
    return rl1



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params:')
for key, val in trial.params.items():
    print(f'    {key}: {val}')
    
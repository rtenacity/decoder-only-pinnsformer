import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import scipy.io
import optuna
from model_components.util import *
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
u_train = torch.tensor(u_train, dtype=torch.float32, requires_grad=True).to(device)
v_train = torch.tensor(v_train, dtype=torch.float32, requires_grad=True).to(device)

snap = np.array([100])
x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = U_star[:,0,snap]
v_star = U_star[:,1,snap]
p_star = P_star[:,snap]

x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(device)
y_star = torch.tensor(y_star, dtype=torch.float32, requires_grad=True).to(device)
t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True).to(device)


smallest_rl1 = 1e10


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

    def forward(self, x, y, t):
        src = torch.cat((x,y,t), dim=-1)
        return self.linear(src)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      
    
def objective(trial):  
    global x_train, y_train, t_train, u_train, v_train, x_star, y_star, t_star, u_star, v_star, p_star, device, smallest_rl1 
    
    d_hidden = trial.suggest_categorical('d_hidden', [256, 512, 768])
    num_layer = trial.suggest_int('num_layer', 4, 6)

    model = PINNs(in_dim=3, hidden_dim=d_hidden, out_dim=2, num_layer=num_layer).to(device)
    model.apply(init_weights)
    optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

    n_params = get_n_params(model)

    loss_track = []

    for i in range(1000):
        def closure():
            psi_and_p = model(x_train, y_train, t_train)
            psi = psi_and_p[:,0:1]
            p = psi_and_p[:,1:2]

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

            loss = torch.mean((u - u_train)**2) + torch.mean((v - v_train)**2) + torch.mean(f_u**2) + torch.mean(f_v**2)

            loss_track.append(loss.item())

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
    psi = psi_and_p[:,0:1]
    p_pred = psi_and_p[:,1:2]

    u_pred = torch.autograd.grad(psi, x_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
    v_pred = - torch.autograd.grad(psi, y_star, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

    u_pred = u_pred.cpu().detach().numpy()
    v_pred = v_pred.cpu().detach().numpy()
    p_pred = p_pred.cpu().detach().numpy()

    C_opt = find_optimal_pressure_correction(p_star, p_pred)
    print("Optimal constant C to add:", C_opt)
    p_pred = p_pred + C_opt


    rl1 = np.linalg.norm(p_star-p_pred,1)/np.linalg.norm(p_star,1)
    
    if rl1 < smallest_rl1:
        smallest_rl1 = rl1
        print(f"New best model found with RL1: {smallest_rl1}")
        torch.save(model.state_dict(), f'saves/conv-pinn-{trial.number}.pth')
        
    return rl1


print(device)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params:')
for key, val in trial.params.items():
    print(f'    {key}: {val}')
    
    
import numpy as np
import torch
import torch.nn as nn
import random
import optuna
from torch.optim import LBFGS, Adam
from model_components.models import FourierPINNsformer
from model_components.util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reproducibility
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# analytical solution
def h(x):
    return np.exp(- (x - np.pi)**2 / (2 * (np.pi/4)**2))

def u_ana(x, t):
    return h(x) * np.exp(5*t) / (h(x) * np.exp(5*t) + 1 - h(x))


res, b_left, b_right, b_upper, b_lower = get_data([0, 2*np.pi], [0,1], 51, 51)
res_test, _, _, _, _ = get_data([0, 2*np.pi], [0,1], 101, 101)

res = make_time_sequence(res, num_step=5, step=1e-4)
b_left = make_time_sequence(b_left, num_step=5, step=1e-4)
b_right = make_time_sequence(b_right, num_step=5, step=1e-4)
b_upper = make_time_sequence(b_upper, num_step=5, step=1e-4)
b_lower = make_time_sequence(b_lower, num_step=5, step=1e-4)

tensors = [res, b_left, b_right, b_upper, b_lower]
tensors = [torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device) for x in tensors]
res, b_left, b_right, b_upper, b_lower = tensors

x_res, t_res = res[:,:,0:1], res[:,:,1:2]
x_left, t_left = b_left[:,:,0:1], b_left[:,:,1:2]
x_right, t_right = b_right[:,:,0:1], b_right[:,:,1:2]
x_upper, t_upper = b_upper[:,:,0:1], b_upper[:,:,1:2]
x_lower, t_lower = b_lower[:,:,0:1], b_lower[:,:,1:2]

# test grid and true solution
u_true = u_ana(res_test[:,0], res_test[:,1]).reshape(101,101)

res_test = make_time_sequence(res_test, num_step=5, step=1e-4)
res_test = torch.tensor(res_test, dtype=torch.float32).to(device)
x_test, t_test = res_test[:,:,0:1], res_test[:,:,1:2]


kernel_size = 300

D1 = kernel_size
D2 = len(x_left)
D3 = len(x_lower)


def compute_ntk(J1, J2):
    Ker = torch.matmul(J1, torch.transpose(J2, 0, 1))
    return Ker

smallest_rl1 = 1e10  # Initialize a large value to track the smallest L1 error

# objective for optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    global device, x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower, x_test, t_test, u_true, D1, D2, D3, kernel_size, smallest_rl1
    set_seed(0)

    # sample hyperparameters
    d_hidden = trial.suggest_categorical('d_hidden', [128, 256, 512, 768])
    d_model = trial.suggest_categorical('d_model', [16, 32, 64, 128])
    mapping_size = trial.suggest_int('mapping_size', 16, 128, step=16)

    # build model
    model = FourierPINNsformer(
        d_out=1,
        d_hidden=d_hidden,
        d_model=d_model,
        N=1,
        heads=2,
        d_in=2,
        mapping_size=mapping_size,
        x_range=(0.0, 2*torch.pi),
        t_range=(0.0, 1.0),
        activation_function='wave_act'
    ).to(device)

    # init weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    
    n_params = get_n_params(model)

    # load data

    # training loop
    w1, w2, w3 = 1, 1, 1

    pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).to(device)

    for i in range(500):
        if i % 20 == 0:
            J1 = torch.zeros((D1, n_params))
            J2 = torch.zeros((D2, n_params))
            J3 = torch.zeros((D3, n_params))

            batch_ind = np.random.choice(len(x_res), kernel_size, replace=False)
            x_train, t_train = x_res[batch_ind], t_res[batch_ind]

            pred_res = model(x_train, t_train)
            pred_left = model(x_left, t_left)
            pred_upper = model(x_upper, t_upper)
            pred_lower = model(x_lower, t_lower)

            for j in range(len(x_train)):
                model.zero_grad()
                pred_res[j,0].backward(retain_graph=True)
                J1[j, :] = torch.cat([
                    p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) 
                    for p in model.parameters()
                    ])


            for j in range(len(x_left)):
                model.zero_grad()
                pred_left[j,0].backward(retain_graph=True)
                J2[j, :] = torch.cat([
                    p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) 
                    for p in model.parameters()
                    ])

            for j in range(len(x_lower)):
                model.zero_grad()
                pred_lower[j,0].backward(retain_graph=True)
                pred_upper[j,0].backward(retain_graph=True)
                J3[j, :] = torch.cat([
                    p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) 
                    for p in model.parameters()
                    ])

            K1 = torch.trace(compute_ntk(J1, J1))
            K2 = torch.trace(compute_ntk(J2, J2))
            K3 = torch.trace(compute_ntk(J3, J3))
            
            K = K1+K2+K3

            w1 = K.item() / K1.item()
            w2 = K.item() / K2.item()
            w3 = K.item() / K3.item()

        def closure():
            pred_res = model(x_res, t_res)
            pred_left = model(x_left, t_left)
            pred_right = model(x_right, t_right)
            pred_upper = model(x_upper, t_upper)
            pred_lower = model(x_lower, t_lower)

            u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]

            loss_res = torch.mean((u_t - 5 * pred_res * (1-pred_res)) ** 2)
            loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
            loss_ic = torch.mean((pred_left[:,0] - torch.exp(- (x_left[:,0] - torch.pi)**2 / (2*(torch.pi/4)**2))) ** 2)


            loss = w1 * loss_res + w3 * loss_bc + w2 * loss_ic
            optim.zero_grad()
            loss.backward()
            return loss
        
        optim.step(closure)
    # evaluate L1 error
    model.eval()
    with torch.no_grad():
        pred = model(x_test, t_test)[:,0].cpu().numpy().reshape(101,101)
    rl1 = np.sum(np.abs(u_true - pred)) / np.sum(np.abs(u_true))
    
    if rl1 < smallest_rl1:
        smallest_rl1 = rl1
        print(f'New smallest L1 error: {smallest_rl1:.4f} at trial {trial.number}')
        torch.save(model.state_dict(), f'saves/1drxn_spformer_trial_{trial.number}.pth')

    return rl1

# run study


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params:')
for key, val in trial.params.items():
    print(f'    {key}: {val}')
    
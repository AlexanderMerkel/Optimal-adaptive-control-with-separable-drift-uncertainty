# The code uses the same structure as the DGP.py file, and calculates the cost using the Feynman Kaz representation of the cost functional.

import torch as th, time, matplotlib.pyplot as plt
import torch.nn as nn
d = 4

if th.cuda.is_available():
    print(th.cuda.device_count(),'CUDAs available')
device = th.device("cuda:0")
th.set_default_tensor_type('torch.cuda.FloatTensor')
th.set_default_dtype(th.float32)

m_V = 512; n_V = 2

class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, L):
        super(CustomNeuralNetwork, self).__init__()
        self.L = L
        self.sigmoid = nn.Sigmoid()
        
        self.W1 = nn.Linear(input_size, hidden_size)
        self.Uz = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wz = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(L)])
        self.Ug = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wg = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wr = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(L)])
        self.Uh = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wh = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(L)])
        
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        S = self.sigmoid(self.W1(x))
        for l in range(self.L):
            Z = self.sigmoid(self.Uz[l](x) + self.Wz[l](S))
            G = self.sigmoid(self.Ug[l](x) + self.Wg(S))
            R = self.sigmoid(self.Ur[l](x) + self.Wr[l](S))
            H = self.sigmoid(self.Uh[l](x) + self.Wh[l](S * R))
            S = (1 - G) * H + Z * S
        
        out = self.output(S)
        return out
    
neural_V = CustomNeuralNetwork(d, m_V, 1, n_V)

y_min = -3; y_max = 3; m_min = -5.0; m_max = 5.0; g_min = 0.0; g_max = 10.0
y0 = 0; T = 1; sig = 1; c = 2; rho = 2; C = 5

dl = 1.0e-4

def density(l):
    return 1.0

def F_m(x, l):
    return l*th.exp(l*x[:,0]-0.5*l**2*x[:,1]) * density(l)

def F(x, l):
    return th.exp(l*x[:,0]-0.5*l**2*x[:,1]) * density(l)

def G(x):
    l = th.arange(0, 1, dl)
    integral1 = th.trapezoid(F_m(x.unsqueeze(-1), l), l)
    integral2 = th.trapezoid(F(x.unsqueeze(-1), l), l)
    return integral1 / integral2

Riccati_Solution = th.load('outputs/FD_Riccati.pt').to(device)
dt = 1.0e-4
dlambda = 1.0e-4

def Riccati(t, l):
    index_lam = th.floor_divide(l, dlambda).int()
    index_t = th.floor_divide(t, dt).int()
    return Riccati_Solution[index_lam, index_t]

def u_LQR(x, lam):
    return -(lam / rho) * Riccati(x[:, 0], lam) * x[:, 1]

naive_lam = th.tensor(0.5)
def HJB_operator(u, x): 
    Du = th.autograd.grad(u(x).sum(dim=0), x, create_graph=True)[0]
    Dyu = (Du @ th.eye(d)[1]); Dmu = (Du @ th.eye(d)[2])
    
    Dtu, Dyu, Dmu, Dgu = Du.unbind(1)
    Dyyu, Dymu = th.autograd.grad(Dyu.sum(dim=0), x, create_graph=True)[0][:,1:3].unbind(1)
    Dmmu = th.autograd.grad(Dmu.sum(dim=0), x, create_graph=True)[0][:,2]
    
    G_out = G(x[:, 2:4])
    ctrl_eval = u_LQR(x, naive_lam)
    
    term1 = ctrl_eval * (G_out * Dyu + Dymu) + ctrl_eval**2 * (((1/sig**2) * G_out *  Dmu + (1/sig**2) * Dgu + (1/sig**2) * Dmmu/2 + rho))
    term3 = Dtu + sig ** 2 * Dyyu / 2 + th.tensor(c) * x[:, 1] ** 2
    
    return term1 + term3

def terminal(x):
    return C * x[:,1] ** 2

min_point = th.tensor([0, y_min, m_min, g_min])
max_point = th.tensor([T, y_max, m_max, g_max])

def eta(x):
    return (T - x[:, 0])

def loss_embed(u, sample):
    def u_composed(x):
        return u(x) * eta(x).unsqueeze(1) + terminal(x).unsqueeze(1)
    loss_int = HJB_operator(u_composed, sample)
    return th.mean(loss_int**2)

opt_V = th.optim.Adam(neural_V.parameters(), lr=1.0e-3)
dist = th.distributions.Uniform(min_point, max_point)
scheduler_V = th.optim.lr_scheduler.MultiStepLR(opt_V, milestones=[8000, 13000], gamma=0.1)

def train(batch_size, neural_V, opt_V):
    l_V = th.inf; i = 0
    loss_history = []
    while l_V > 1.0e-4 and i<30000:
        opt_V.zero_grad()
        sample = dist.sample((batch_size,)).requires_grad_(True)
        l_V = loss_embed(neural_V, sample)
        l_V.backward()
        opt_V.step()
        scheduler_V.step()
        i += 1
        if i % 10 == 0:
            print('epoch:', i,'Loss V: ', "{:.2e}".format(l_V.item()))
    return loss_history

batch_size = 10000
loss_history = train(batch_size, neural_V, opt_V)

th.save(neural_V.state_dict(), 'outputs/neural_V_naive.pt')
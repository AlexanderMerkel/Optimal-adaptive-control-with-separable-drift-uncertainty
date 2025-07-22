# %%
import os
import torch as th
import torch.nn as nn

d = 4 # Dimension of the state space

if th.cuda.is_available():
    print(th.cuda.device_count(),'CUDAs available')
    device = th.device("cuda:0")
    th.set_default_tensor_type('torch.cuda.FloatTensor')
th.set_default_dtype(th.float16)

# Number of hidden layers and nodes for the neural network approximators
# m_V and n_V for the value function approximator (higher number of parameters)
# m_ctrl and n_ctrl for the control function approximator
m_V = 32; n_V = 2

# The neural network approximator for the value function / PDE solution. It has the DGM architecture.
class CustomNeuralNetwork(nn.Module):
    """A neural network with the DGM architecture for the value function / PDE solution.

    Args:
        nn (nn.Module): PyTorch neural network module.
    """
    def __init__(self, input_size, hidden_size, output_size, L):
        """Initializes the neural network with the DGM architecture.

        Args:
            input_size (int): The dimension of the input, usually the state space dimension.
            hidden_size (int): The Number of hidden nodes in the DGM layers.
            output_size (int): The output dimension of the neural network, usually 1 for the value function / PDE solution.
            L (int): The number of DGM layers.
        """
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

dl = 1.0e-3 # Resolution of the integral calculation in the trapzoid rule.

def density(l):
    """The density function of the hidden random variable lambda in a vectorized form.

    Args:
        l (torch.float32): The value of the hidden random variable lambda.

    Returns:
        torch.float32: The density function of the hidden random variable lambda.
    """
    return 1.0

#The coefficient functions.
def F_m(x, l):
    return l*th.exp(l*x[:,0]-0.5*l**2*x[:,1]) * density(l)

def F(x, l):
    return th.exp(l*x[:,0]-0.5*l**2*x[:,1]) * density(l)

def G(x):
    """The function G appearing in the HJB operator, which is the L2-estimator of lambda given x.

    Args:
        x (torch.Tensor): A tensor with values of the auxiliary states.

    Returns:
        _type_: _description_
    """
    l = th.arange(0, 1, dl) #Lambda takes values in [0,1] and is ~Uniform(0,1)
    integral1 = th.trapezoid(F_m(x.unsqueeze(-1), l), l)
    integral2 = th.trapezoid(F(x.unsqueeze(-1), l), l)
    return integral1 / integral2
# The HJB operator. It takes as input the value function / PDE solution u, the state x and the control / minimizer of the HJB operator ctrl. It is implemented to not compute the full Hessians, but only the necessary derivatives.
def HJB_operator(u, x):
    Du = th.autograd.grad(u(x).sum(dim=0), x, create_graph=True)[0]
    Dyu = (Du @ th.eye(d)[1]); Dmu = (Du @ th.eye(d)[2])
    
    Dtu, Dyu, Dmu, Dgu = Du.unbind(1)
    Dyyu, Dymu = th.autograd.grad(Dyu.sum(dim=0), x, create_graph=True)[0][:,1:3].unbind(1)
    Dmmu = th.autograd.grad(Dmu.sum(dim=0), x, create_graph=True)[0][:,2]
    
    G_out = G(x[:, 2:4])
    term1 = (G_out * Dyu + Dymu) / ((((1/sig**2) * G_out *  Dmu + (1/sig**2) * Dgu + (1/sig**2) * Dmmu/2 + rho))
)
    term3 = Dtu + sig ** 2 * Dyyu / 2 + th.tensor(c) * x[:, 1] ** 2
    
    return term1 + term3
# The terminal condition of the PDE.
def terminal(x):
    return C * x[:,1] ** 2 
# The domain on which the PDE is solved. As pointed out in the paper, it is useful to solve the PDE on a larger domain than the one of interest, to avoid boundary effects, but also not too large, to avoid unnecessary computations. We obtained an a priori estimate on the domain by using Deep Reinforcement Learning with policy gradient for a rough estimate on the domain size required.

y_min = -3; y_max = 3; m_min = -5.0; m_max = 5.0; g_min = 0.0; g_max = 10.0
y0 = 0; T = 1; sig = 1.0; c = 2; rho = 2; C = 5

min_point = th.tensor([0, y_min, m_min, g_min])
max_point = th.tensor([T, y_max, m_max, g_max])

def eta(x):
    return (T - x[:, 0]) 
# We embed the boundary condition into the approximator, that is we do not approximate the value function directly using the neural network, but a transformation: u(x) = V(x) * eta(x) + terminal(x).
def loss_embed(u, sample):
    def u_composed(x): # The composed approximator
        return u(x) * eta(x).unsqueeze(1) + terminal(x).unsqueeze(1)
    loss_int = HJB_operator(u_composed, sample)
    return th.mean(loss_int**2)

# We use the Adam optimizer with a learning rate of 1.0e-3. We use a learning rate scheduler, which reduces the learning rate by a factor of 10 after 8000 and 13000 epochs, found to be empirically good.
opt_V= th.optim.Adam(neural_V.parameters(), lr=1.0e-3)
dist = th.distributions.Uniform(min_point, max_point)
scheduler_V = th.optim.lr_scheduler.MultiStepLR(opt_V, milestones=[8000, 13000], gamma=0.1)

def train(batch_size, neural_V, opt_V,):
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
        print(l_V.item())
        if i % 10 == 0:
            print('epoch:', i,'Loss V: ', "{:.2e}".format(l_V.item()))
            loss_history.append(l_V.item())
    return loss_history

# %%
batch_size = 100 
loss_history = train(batch_size, neural_V, opt_V)

if not os.path.exists('outputs'):
    os.makedirs('outputs')

th.save(neural_V.state_dict(), 'outputs/neural_V.pt')

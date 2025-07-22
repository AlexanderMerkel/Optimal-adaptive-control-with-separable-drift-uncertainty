# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set device and default tensor type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)  # Change back to float32
print('Using device:', device)

# Create output directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Constants (model parameters)
lambda_h = 2.0   # High value of lambda
lambda_l = -2.0   # Low value of lambda
q_lh = 0.2       # Transition rate from low to high
q_hl = 0.1       # Transition rate from high to low
k = 1.0          # Model parameter
rho = 0.5        # Positive constant
delta = 0.01     # Discount rate, positive constant

class DGMNet(nn.Module):
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
        super(DGMNet, self).__init__()
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
# Initialize network
input_dim = 2
hidden_dim = 8
output_dim = 1
num_layers = 2

net = DGMNet(input_dim, hidden_dim, output_dim, num_layers).to(device)  # Remove .to(torch.float16)
print('Net initialized with', sum(p.numel() for p in net.parameters()), 'parameters')

# Define the PDE residual function
def pde_residual(y : torch.Tensor, p : torch.Tensor):
    y.requires_grad = True
    p.requires_grad = True
    x = torch.cat([y, p], dim=1)  # Remove .to(torch.float16)
    V = net(x)

    # Compute first derivatives
    V_y = torch.autograd.grad(V, y, grad_outputs=torch.ones_like(V), retain_graph=True, create_graph=True)[0]
    V_p = torch.autograd.grad(V, p, grad_outputs=torch.ones_like(V), retain_graph=True, create_graph=True)[0]

    # Compute second derivatives
    V_yy = torch.autograd.grad(V_y, y, grad_outputs=torch.ones_like(V_y), retain_graph=True, create_graph=True)[0]
    V_yp = torch.autograd.grad(V_y, p, grad_outputs=torch.ones_like(V_y), retain_graph=True, create_graph=True)[0]
    V_pp = torch.autograd.grad(V_p, p, grad_outputs=torch.ones_like(V_p), retain_graph=True, create_graph=True)[0]

    if torch.isnan(V_y).any() or torch.isnan(V_p).any() or torch.isnan(V_yy).any() or torch.isnan(V_yp).any() or torch.isnan(V_pp).any():
        print(V_y, V_p, V_yy, V_yp, V_pp)
        raise ValueError('NaN encountered in gradients')

    # Compute the PDE residual
    # First compute the numerator and denominator of the first term
    numerator = ((lambda_h * p + lambda_l * (1 - p)) * V_y + p * (1 - p) * (lambda_h - lambda_l) * V_yp)
    numerator = numerator.squeeze(1)
    numerator = numerator ** 2

    denominator = p ** 2 * (1 - p) ** 2 * (lambda_h - lambda_l) ** 2 * V_pp + rho
    denominator = denominator.squeeze(1)

    if torch.any(denominator < 1e-6):
        raise ValueError('Denominator is very small')

    # To prevent division by zero or negative denominator, add a small epsilon
    epsilon = 1e-6
    denominator = torch.where(denominator >= epsilon, denominator, epsilon * torch.ones_like(denominator))

    first_term = -0.5 * numerator / denominator

    # Second term
    second_term = (-q_lh * p + q_hl * (1 - p)) * V_p
    second_term = second_term.squeeze(1)

    # Third term
    third_term = 0.5 * V_yy.squeeze(1)

    # Fourth term
    fourth_term = 0.5 * k * y.squeeze(1) ** 2

    # Fifth term
    fifth_term = -delta * V.squeeze(1)

    residual = first_term + second_term + third_term + fourth_term + fifth_term

    return residual

# Define the loss function
def loss_function(y : torch.Tensor, p : torch.Tensor):
    residual = pde_residual(y, p)
    loss = torch.mean(residual ** 2)
    return loss

# Define the domain
y_min = -5.0
y_max = 5.0
p_min = 1e-2
p_max = 1 - 1e-2

def sample_x(batch_size):
    y = torch.FloatTensor(batch_size, 1).uniform_(y_min, y_max).to(device)
    p = torch.FloatTensor(batch_size, 1).uniform_(0.0, 1.0).to(device)
    return y, p

# Training parameters
optimizer = optim.Adam(net.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8000, 13000], gamma=0.1)
batch_size = 256
num_epochs = 10000

loss_history = []

# Training loop
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    y, p = sample_x(batch_size)
    loss = loss_function(y,p)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4e}')

# %%
# Plot the value function net
plt.figure()
plt.plot(net(sample_x(10000)).detach().cpu().numpy())

# Save the trained model
torch.save(net.state_dict(), 'outputs/dgm_model.pt')



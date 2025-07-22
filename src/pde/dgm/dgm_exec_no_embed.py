# %%
import torch
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print('Using device:', device)

class DGMNet(nn.Module):
    """A neural network with the DGM architecture for the value function / PDE solution.

    Args:
        nn (nn.Module): PyTorch neural network module.
    """
    def __init__(self, input_size: int,
                 hidden_size:int,
                 output_size:int,
                 L:int):
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
        self.bn = nn.BatchNorm1d(hidden_size)
        self.Uz = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wz = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(L)])
        self.Ug = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wg = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wr = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(L)])
        self.Uh = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(L)])
        self.Wh = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(L)])
        
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.sigmoid(self.W1(x))
        for l in range(self.L):
            Z = self.sigmoid(self.Uz[l](x) + self.Wz[l](S))
            G = self.sigmoid(self.Ug[l](x) + self.Wg(S))
            R = self.sigmoid(self.Ur[l](x) + self.Wr[l](S))
            H = self.sigmoid(self.Uh[l](x) + self.Wh[l](S * R))
            S = (1 - G) * H + Z * S
        
        out = self.output(S)
        return out

def pde_residual(z: torch.Tensor,
                    value_net: DGMNet,
                    policy_net: DGMNet) -> torch.Tensor:
    """
    Computes the PDE residual of the given HJB equation:
    
    0 = V_t
        + 1 / [4( rho - p^2(1-p)^2 (lambda_l - lambda_h)^2 V_pp )]
          * [ ( (lambda_l p + lambda_h (1-p)) V_s
                 - V_x + V_{a^l} + V_{a^h}
                 + sigma p(1-p)(lambda_l - lambda_h) V_{sp}
                 + s )
               - 2 p^2(1-p)^2 (lambda_l - lambda_h)
                 (lambda_l kappa_l a^l - lambda_h kappa_h a^h) V_pp
             ]^2
        + [ - lambda_l kappa_l a^l p - lambda_h kappa_h a^h (1-p) ] V_s
        - kappa_l a^l V_{a^l}
        - kappa_h a^h V_{a^h}
        + 1/2 sigma^2 V_{ss}
        + 1/2 p^2(1-p)^2 (lambda_h kappa_h a^h - lambda_l kappa_l a^l)^2 V_{pp}
        + sigma p(1-p) (lambda_h kappa_h a^h - lambda_l kappa_l a^l) V_{sp}
        - c x^2

    In this code snippet:
      - net is assumed to be a global DGM/PyTorch model computing V(s,x,p,a_l,a_h).
      - s,x,p,a_l,a_h all have requires_grad=True for derivative tracking.
      - The PDE residual is returned; a perfect solution means residual=0 everywhere.
    """

    z.requires_grad_(True)

    t: torch.Tensor = z[:, 0:1]
    s: torch.Tensor = z[:, 1:2]
    x: torch.Tensor = z[:, 2:3]
    p: torch.Tensor = z[:, 3:4]
    a_l: torch.Tensor = z[:, 4:5]
    a_h: torch.Tensor = z[:, 5:6]

    inp: torch.Tensor = torch.cat([t, s, x, p, a_l, a_h], dim=1)
    V:torch.Tensor = value_net(inp)
    u:torch.Tensor = policy_net(inp)

    # --------------------------
    # 1) Compute first derivatives
    # --------------------------
    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                retain_graph=True, create_graph=True)[0]
    V_s = torch.autograd.grad(V, s, grad_outputs=torch.ones_like(V),
                              retain_graph=True, create_graph=True)[0]
    V_x = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V),
                              retain_graph=True, create_graph=True)[0]
    V_p = torch.autograd.grad(V, p, grad_outputs=torch.ones_like(V),
                              retain_graph=True, create_graph=True)[0]
    V_a_l = torch.autograd.grad(V, a_l, grad_outputs=torch.ones_like(V),
                                retain_graph=True, create_graph=True)[0]
    V_a_h = torch.autograd.grad(V, a_h, grad_outputs=torch.ones_like(V),
                                retain_graph=True, create_graph=True)[0]

    # --------------------------
    # 2) Compute second derivatives
    # --------------------------
    V_ss = torch.autograd.grad(V_s, s, grad_outputs=torch.ones_like(V_s),
                               retain_graph=True, create_graph=True)[0]
    V_sp = torch.autograd.grad(V_s, p, grad_outputs=torch.ones_like(V_s),
                               retain_graph=True, create_graph=True)[0]
    V_pp = torch.autograd.grad(V_p, p, grad_outputs=torch.ones_like(V_p),
                               retain_graph=True, create_graph=True)[0]

    # --------------------------
    # 3) Define constants/params used in PDE
    # --------------------------
    sigma: float = 1.0 
    rho: float = 1 
    lambda_l: float = 0.5
    lambda_h: float =  0.1
    kappa_l: float = 0.005
    kappa_h: float = 0.2
    c :float = 0.1     # example

    # ------------------------------------------------------------------
    # Build each PDE term
    # ------------------------------------------------------------------
    # 1) Terms multiplying V_s
    drift_term_s = (
        lambda_l * (u - kappa_l * a_l) * p
        + lambda_h * (u - kappa_h * a_h) * (1.0 - p)
    ) * V_s

    # 2) Terms multiplying V_x
    drift_term_x = -u * V_x

    # 3) Terms multiplying V_{a^l} and V_{a^h}
    drift_term_al = (u - kappa_l * a_l) * V_a_l
    drift_term_ah = (u - kappa_h * a_h) * V_a_h

    # 4) The Brownian diffusion in s
    diffusion_s = 0.5 * (sigma**2) * V_ss

    # 5) The second derivative in p with p^2(1-p)^2 [..]^2 factor
    impact_diff = p**2 * (1.0 - p)**2 * (
        lambda_l * (u - kappa_l * a_l) - lambda_h * (u - kappa_h * a_h)
    ) ** 2 * V_pp

    # 6) The cross derivative V_sp
    cross_sp = (
        sigma
        * p
        * (1.0 - p)
        * (lambda_l * (u - kappa_l * a_l) - lambda_h * (u - kappa_h * a_h))
        * V_sp
    )

    # 7) The “(s - rho u) u” part
    control_payoff = (s - rho * u) * u

    # 8) The inventory cost term
    inventory_cost = -c * x**2

    # 9) Summation + V_t
    pde = (
        V_t
        + drift_term_s
        + drift_term_x
        + drift_term_al
        + drift_term_ah
        + diffusion_s
        + impact_diff
        + cross_sp
        + control_payoff
        + inventory_cost
    )

    return pde.squeeze(1)

# Define the loss function
def loss_function(V_net: DGMNet, ctrl_net: DGMNet,
                    z: torch.Tensor, z_bd: torch.Tensor) -> torch.Tensor:
    residual: torch.Tensor = pde_residual(z, V_net, ctrl_net)
    ctrl_loss = residual.mean()
    loss_int: torch.Tensor = torch.mean(residual ** 2)
    s: torch.Tensor = z_bd[:, 1:2]
    x: torch.Tensor = z_bd[:, 2:3]
    C: float = 5.0
    loss_bd: torch.Tensor = torch.mean((V_net(z_bd) - (s * x + C * x**2))**2)
    return loss_int + loss_bd, -ctrl_loss

# Define the domain
t_min, t_max = 0.0, 11.0
s_min, s_max = 0.0, 2.0
x_min, x_max = 0.0, 11.0
p_min, p_max = 0.0, 1.0
a_l_min, a_l_max = 0.0, 1.0
a_h_min, a_h_max = 0.0, 1.0

def sample_x(batch_size):
    t = torch.FloatTensor(batch_size, 1).uniform_(t_min, t_max).to(device)
    s = torch.FloatTensor(batch_size, 1).uniform_(s_min, s_max).to(device)
    x = torch.FloatTensor(batch_size, 1).uniform_(x_min, x_max).to(device)
    p = torch.FloatTensor(batch_size, 1).uniform_(p_min, p_max).to(device)
    a_l = torch.FloatTensor(batch_size, 1).uniform_(a_l_min, a_l_max).to(device)
    a_h = torch.FloatTensor(batch_size, 1).uniform_(a_h_min, a_h_max).to(device)
    return torch.cat([t, s, x, p, a_l, a_h], dim=1)

# Initialize network
input_dim = 6
hidden_dim = 64
output_dim = 1
num_layers = 2

V_net = DGMNet(input_dim, hidden_dim, output_dim, num_layers).to(device)
ctrl_net = DGMNet(input_dim, hidden_dim, output_dim, num_layers).to(device)
print('V_Net initialized with', sum(p.numel() for p in V_net.parameters()), 'parameters')
print('Ctrl_Net initialized with', sum(p.numel() for p in ctrl_net.parameters()), 'parameters')

# Training parameters
V_optimizer = optim.Adam(V_net.parameters(), lr=1e-3)
ctrl_optimizer = optim.Adam(V_net.parameters(), lr=1e-3)
V_scheduler = optim.lr_scheduler.MultiStepLR(V_optimizer,
                                           milestones=[5000, 10000, 15000], gamma=0.5)
ctrl_scheduler = optim.lr_scheduler.MultiStepLR(ctrl_optimizer,
                                           milestones=[5000, 10000, 15000], gamma=0.5)
batch_size = 1500
num_epochs = 10000

loss_history = []

def train(num_epochs, batch_size):
    for epoch in range(1, num_epochs + 1):
        V_optimizer.zero_grad()
        ctrl_optimizer.zero_grad()
        z = sample_x(batch_size)
        z_bd = sample_x(batch_size//6)
        z_bd[:, 0] = t_max
        V_loss, ctrl_loss = loss_function(V_net, ctrl_net, z, z_bd)
        if epoch % 50 == 0:
            V_loss.backward(retain_graph=True)
            V_optimizer.step()
            V_scheduler.step()

        if epoch % 50 != 0:
            ctrl_loss.backward()
            ctrl_optimizer.step()
            ctrl_scheduler.step()

        loss_history.append(V_loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch: {epoch}, V_Loss: {V_loss.item():.4e}, ctrl_loss:{ctrl_loss.item():.4e}')


    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    torch.save(V_net.state_dict(), 'outputs/V_dgm_model.pt')
    torch.save(ctrl_net.state_dict(), 'outputs/ctrl_dgm_model.pt')

if __name__ == '__main__':
    train(num_epochs, batch_size)
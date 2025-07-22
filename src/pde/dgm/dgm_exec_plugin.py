# %%
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print('Using device:', device)

# Create output directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

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
input_dim = 6
hidden_dim = 256
output_dim = 1
num_layers = 6

net = DGMNet(input_dim, hidden_dim, output_dim, num_layers).to(device)  # Remove .to(torch.float16)
print('Net initialized with', sum(p.numel() for p in net.parameters()), 'parameters')

# Define the PDE residual function
def pde_residual(t: torch.Tensor,
                s: torch.Tensor,
                x: torch.Tensor,
                p: torch.Tensor,
                a_l: torch.Tensor,
                a_h: torch.Tensor) -> torch.Tensor:
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


    def approximator(t: torch.Tensor,
                        s: torch.Tensor,
                        x: torch.Tensor,
                     V_inter: torch.Tensor) -> torch.Tensor:
        """We do not directly use the DGM network to compute the value function, but embed the boundary condition."""
        def eta(t: torch.Tensor) -> torch.Tensor:
            return (T - t)
        C: float = 5  # example
        T: float = 10  # example
        boundary = s * x + C * x**2
        approximator = V_inter * eta(t) + boundary
        return approximator 

    # Ensure gradients can flow
    for var in (t, s, x, p, a_l, a_h):
        var.requires_grad_(True)

    # Concatenate inputs to feed the network: (s, x, p, a_l, a_h)
    inp = torch.cat([t, s, x, p, a_l, a_h], dim=1)
    V_inter = net(inp)       # Forward pass: V(t, s, x, p, a_l, a_h)
    V = approximator(t, s, x, V_inter)  # Embed boundary condition

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

    # --------------------------
    # 4) Build each PDE term
    # --------------------------
    # 4a) Big fraction coefficient denominator for the squared bracket
    denom = 4.0 * (rho - p**2 * (1-p)**2 * (lambda_l - lambda_h)**2 * V_pp)

    # 4b) Bracket that gets squared
    bracket = ((lambda_l * p + lambda_h * (1 - p)) * V_s
               - V_x + V_a_l + V_a_h
               + sigma * p * (1 - p) * (lambda_l - lambda_h) * V_sp
               + s
               - 2.0 * p**2 * (1 - p)**2 * (lambda_l - lambda_h)
                 * (lambda_l * kappa_l * a_l - lambda_h * kappa_h * a_h) * V_pp)

    # 4c) Term outside the fraction
    #    => (1 / denom) * (bracket^2)
    frac_term = bracket**2 / denom

    # 4d) Additional PDE parts
    part2 = ((-lambda_l * kappa_l * a_l * p
              - lambda_h * kappa_h * a_h * (1 - p)) * V_s)
    part3 = (- kappa_l * a_l * V_a_l - kappa_h * a_h * V_a_h)
    part4 = 0.5 * sigma**2 * V_ss
    part5 = 0.5 * p**2 * (1-p)**2 * (lambda_h * kappa_h * a_h
                                     - lambda_l * kappa_l * a_l)**2 * V_pp
    part6 = (sigma * p * (1-p) * (lambda_h * kappa_h * a_h
                                  - lambda_l * kappa_l * a_l)
             * V_sp)
    part7 = - c * x**2   # inventory cost or penalty

    # --------------------------
    # 5) PDE residual
    # The PDE states: 0 = V_t + [the sum of all terms].
    # If time T is not a variable in net, then V_t=0 (stationary).
    # We just define PDE = fraction + ... etc.
    # So PDE_res = fraction_term + part2 + part3 + part4 + part5 + part6 + part7
    # Setting PDE_res = 0 means we solve PDE_res**2 -> 0 in a least-squares sense.
    # --------------------------
    # If your PDE is time-dependent and you have 't' input,
    # you'd compute V_t similarly via torch.autograd, then add it to the sum.

    pde = V_t + frac_term + part2 + part3 + part4 + part5 + part6 + part7

    return pde.squeeze(1)  # PDE residual; model trains to make this ~ 0

# Define the loss function
def loss_function(t: torch.Tensor, s:torch.Tensor,
                  x: torch.Tensor, p: torch.Tensor,
                  a_l: torch.Tensor, a_h: torch.Tensor) -> torch.Tensor:
    residual = pde_residual(t, s, x, p, a_l, a_h)
    loss = torch.mean(residual ** 2)
    return loss

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
    return t, s, x, p, a_l, a_h

# Training parameters
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[5000, 10000, 15000],
                                             gamma=0.5)
batch_size = 2048
num_epochs = 10000

loss_history = []

# Training loop
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    t, s, x, p, a_l, a_h = sample_x(batch_size)
    loss = loss_function(t, s, x, p, a_l, a_h)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4e}')

# Plot the value function net
plt.figure()
plt.plot(net(sample_x(10000)).detach().cpu().numpy())

# Save the trained model
torch.save(net.state_dict(), 'outputs/dgm_model.pt')
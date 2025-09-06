# Project Documentation

## Overview

This project focuses on **finite-horizon optimal adaptive control** numerics, implementing various algorithms for solving partially observable stochastic control problems. The main algorithms include:

- **PDE/DGM**: Deep Galerkin Method for solving HJB equations
- **RL/DDPG**: Deep Deterministic Policy Gradient for continuous control
- **RL/REINFORCE**: Policy gradient method for discrete-time optimization

## Directory Structure

- `src/`: Contains all source code files and algorithm implementations
- `tests/`: Includes test-related data and outputs
- `docs/`: Documentation and project notes

## Getting Started

1. Navigate to the `src/` directory to explore the code
2. Refer to the `tests/` directory for test cases and outputs
3. Read the `README.md` and `notes/` in the `docs` folder for detailed information

## Control Problem Formulation

### System Dynamics

Let $T > 0$ be the finite time horizon with discrete time steps $t_k = k\Delta t$, $k=0,...,N$.
For simplicity of notation, we denote the innovations process $I^{u}_t = \rd W_t$.
The augmented state process $(S_t, X_t, p_t, A^l_t, A^h_t)$ follows:

```math
\begin{aligned}
\mathrm{d} S_t &= -\big[\lambda_l(u_t + \kappa_l A^l_t)p_t + \lambda_h(u_t + \kappa_h A^h_t)(1-p_t)\big]\mathrm{d} t +  \sigma \mathrm{d} W_t \\
\mathrm{d} X_t &= -u_t \mathrm{d} t \\
\mathrm{d} p_t &= \frac{1}{\sigma} p_t(1-p_t)\big[\lambda_l(u_t + \kappa_l A^l_t) - \lambda_h(u_t + \kappa_h A^h_t)\big]\mathrm{d} W_t \\
\mathrm{d} A^l_t &= (u_t + \kappa_l A^l_t)\mathrm{d} t \\
\mathrm{d} A^h_t &= (u_t + \kappa_h A^h_t)\mathrm{d} t
\end{aligned}
```

where:

- $S_t$: Observable state process
- $X_t$: Inventory process
- $p_t$: Belief state ($\mathbb{P}(\lambda = \lambda_l|\mathcal{F}_t)$)
- $A^l_t, A^h_t$: Accumulator states
- $W_t$: Brownian motion, $I_t$: Innovation process

### Cost Functional

Maximize the profit from liquidation expected cost over finite horizon $[0,T]$:

```math
J(u) = \mathbb{E}\left[\int_0^T \left((S_t - \rho u_t)u_t - c X_t^2\right)\mathrm{d} t + \left(S_T X_T - C X_T^2\right)\right]
```

with parameters:

- $\rho > 0$: Control penalty
- $c > 0$: Running cost coefficient
- $C > 0$: Terminal cost coefficient

## Numerical Implementation

The augmented state-space formulation enables Markovian control:

```math
\begin{aligned}
\text{State} &= (t, S, X, p, A^l, A^h) \in [0,T] \times \mathbb{R}^+ \times \mathbb{R}^+ \times [0,1]^3 \\
\text{Control} &= u_t \in \mathbb{R}
\end{aligned}
```

### Key Parameters

| Parameter              | Description              | Value                                                      |
| ---------------------- | ------------------------ | ---------------------------------------------------------- |
| $T$                    | Time horizon             | 10                                                         |
| $N$                    | Time steps               | 250                                                        |
| $\rho$                 | Temporary Impact         | 0.5                                                        |
| $\sigma$               | Volatility               | 1.0                                                        |
| $c$                    | Running cost             | 0.1                                                        |
| $C$                    | Terminal cost            | 10                                                         |
| $\lambda_l, \lambda_h$ | Transient Impact         | 1.5, 0.5                                                   |
| $\kappa_l, $\kappa_h$  | Resilience Rates         | 0.5, 1.5                                                   |
| Initial State          | Starting state variables | $t=0.0$, $S=10.0$, $X=10.0$, $p=0.5$, $A^l=0.0$, $A^h=0.0$ |

## Algorithm Implementation

### REINFORCE Components

```python
# Gaussian policy network
class PolicyNetwork(nn.Module):
    def forward(self, x):
        return mean, log_std  # (batch_size, 1)

# Environment step
def step(self, actions):
    # Euler-Maruyama discretization
    dS = drift*dt + sqrt(dt)*dW
    dX = -u*dt
    dp = innovation_update
    return next_state, reward

# Training loop
def train_policy(...):
    # Trajectory rollout
    # Policy gradient update: θ ← θ + α∇[log_probs · returns]
```

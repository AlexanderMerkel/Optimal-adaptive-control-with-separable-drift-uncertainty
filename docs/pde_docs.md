# Project Documentation

## Overview
This project focuses on optimal adaptive control numerics, implementing various algorithms for solving control problems. The main algorithms include:

- **PDE/DGM**: Deep Galerkin Method for solving partial differential equations.
- **RL/DDPG**: Deep Deterministic Policy Gradient for reinforcement learning.
- **RL/REINFORCE**: REINFORCE algorithm for policy optimization.

## Directory Structure

- `src/`: Contains all source code files and algorithm implementations.
- `tests/`: Includes test-related data and outputs.
- `docs/`: Documentation and project notes.

## Getting Started

1. Navigate to the `src/` directory to explore the code.
2. Refer to the `tests/` directory for test cases and outputs.
3. Read the `README.md` and `notes/` in the `docs` folder for detailed information.

The concrete optimal control problem is defined as:

Let $\lambda = (\lambda_{t})_{t\geq0}$ be a two-state Markov chain with $Q$-matrix $Q$ taking
values $\lambda_{l} < \lambda_{h}$, and let $W$ be a standard Brownian motion.

The $Q$-matrix $Q$ is given by

```math
Q = \begin{pmatrix}
    -q_{lh} & q_{lh} \\
    q_{hl} & -q_{hl},
\end{pmatrix}
```

where $q_{lh}$ is the rate of transition from $\lambda_{l}$ to $\lambda_{h}$ and $q_{hl}$ is the rate of transition from $\lambda_{h}$ to $\lambda_{l}$.

The initial distribution of $\lambda$ is given by $\P(\lambda_{0} = \lambda_{l}) = p^{l}_{0}$ and
$\P(\lambda_{0} = \lambda_{h}) = p^{h}_{0} = 1 - p^{l}_{0}$ and $p_{0} = (p^{h}_{0}, p^{l}_{0})$.

Let $u$ be a control and consider the controlled SDE

```math
\mathrm{d} Y^{u}_{t} = \lambda_{t} u_{t} \mathrm{d} t + \mathrm{d} W_{t}.
```

Defining $p_{t} = \P(\lambda_{t} = \lambda_{l} | \mathcal{Y}^{u}_{t})$, we have

```math
\mathrm{d} p^{u}_{t} = \bigl(-q_{lh} p^{u}_{t} + q_{hl} (1 - p^{u}_{t})\bigr) \mathrm{d} t
+ p^{u}_{t}(1 - p^{u}_{t})(\lambda_{h} - \lambda_{l}) u_{t} \mathrm{d} I^{u}_{t},
\quad p^{u}_{0} = p_{0}.
```

### Markovian lift

Hence, the Markovian lift is given by

```math
\begin{aligned}
    \mathrm{d} Y^{u}_{t} &= (\lambda_{h}p^{u}_{t} + \lambda_{l}(1 - p^{u}_{t})) u_{t} \mathrm{d} t + \mathrm{d} I^{u}_{t}, &
    \quad Y^{u}_{0} &= 0 \\
    \mathrm{d} p^{u}_{t} &= \bigl(-q_{lh} p^{u}_{t} + q_{hl} (1 - p^{u}_{t})\bigr) \mathrm{d} t
    + p^{u}_{t}(1 - p^{u}_{t})(\lambda_{h} - \lambda_{l}) u_{t} \mathrm{d} I^{u}_{t}, &
    \quad p^{u}_{0} &= p_{0}.
\end{aligned}
```

The cost functional is given by

```math
J(u) = \mathbb{E}\biggl[\int_{0}^{\infty} e^{-\delta t}
\frac{1}{2} \biggl(k \bigl(Y^{u}_{t}\bigr)^{2} + \rho u_{t}^{2}\biggr) \mathrm{d} t\biggr].
```

On an infinite horizon, LQG problem, the HJB equation is given by

```math
\begin{aligned}
    \sup_{u\in\mathcal{U}}&\biggl\{
    \bigl(\lambda_{h}p + \lambda_{l}(1 - p)\bigr)u V_{y} 
    + \bigl(-q_{lh} p + q_{hl} (1 - p)\bigr)V_{p}\biggr.\\
    \biggl.&+ \frac{1}{2}\bigl( V_{yy} 
    + p^{2}(1 - p)^{2}(\lambda_{h} - \lambda_{l})^{2}u^{2}V_{pp}
    + 2 p(1 - p)(\lambda_{h} - \lambda_{l})uV_{yp} \bigr)\biggr\}
    + \frac{1}{2}\bigl( k y^{2} + \rho u^{2}\bigr)
    - \delta V = 0
\end{aligned}
```

and the minimizer is given by

```math
u^{*} = -\frac{\bigl(\lambda_{h}p + \lambda_{l}(1 - p)\bigr)V_{y} 
+ p(1 - p)(\lambda_{h} - \lambda_{l})V_{yp}}{p^{2}(1 - p)^{2}(\lambda_{h} - \lambda_{l})^{2}V_{pp} + \rho}.
```

Hence, the HJB equation is given by

```math
\begin{aligned}
    &-\frac{1}{2}\frac{\Bigl(\bigl(\lambda_{h}p + \lambda_{l}(1 - p)\bigr)V_{y} 
    + p(1 - p)(\lambda_{h} - \lambda_{l})V_{yp}\Bigr)^{2}}{p^{2}(1 - p)^{2}(\lambda_{h} - \lambda_{l})^{2}V_{pp} + \rho}\\
    &\quad + \bigl(-q_{lh} p + q_{hl} (1 - p)\bigr)V_{p}
    + \frac{1}{2} V_{yy}
    + \frac{1}{2} k y^{2}
    - \delta V = 0
\end{aligned}
```

and $k, \rho, \delta > 0$ are model parameters.

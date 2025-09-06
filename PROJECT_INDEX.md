# Optimal Adaptive Control Numerics - Project Index

## Project Overview

**Research Focus**: Numerical methods for optimal adaptive control with separable drift uncertainty  
**Academic Context**: Implementation companion for "Optimal adaptive control with separable drift uncertainty" ([arXiv:2309.07091](https://arxiv.org/abs/2309.07091))  
**Core Problem**: Optimal execution with transient and permanent price impact under regime uncertainty

## Quick Navigation

- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Implementation Methods](#implementation-methods)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Documentation](#documentation)
- [Mathematical Foundation](#mathematical-foundation)

## Project Structure

```
optimal_adaptive_control_numerics/
├── src/                           # Core implementation
│   ├── control_theory/           # Main framework package
│   │   ├── __init__.py          # Package exports and API
│   │   ├── config.py            # Configuration management
│   │   ├── environment.py       # Simulation environment
│   │   ├── policies.py          # Policy implementations
│   │   ├── reinforce_agent.py   # REINFORCE RL agent
│   │   ├── riccati_policies.py  # LQR/Riccati solutions
│   │   ├── hjb_solver.py        # HJB PDE solver
│   │   └── comparison.py        # Performance comparison tools
│   ├── model_parameters.yaml    # Central configuration
│   └── model_parameters_test.yaml
├── docs/                         # Documentation
│   ├── README.md                # Project description
│   ├── pde_docs.md              # PDE method documentation
│   ├── rl_docs.md               # RL method documentation
│   └── notes/                   # Research notes
├── control_comparison_standalone.ipynb  # Standalone comparison
├── literature.md                # Literature review
├── requirements.txt             # Dependencies
└── Optimal Execution with Unobservable Price Impact/  # LaTeX paper
```

## Core Components

### 1. Problem Configuration (`config.py`)

- **OptimalExecutionConfig**: Central problem parameters
- **Default Configuration**: Standard academic benchmark setup
- **State Space Definition**: 6D state `(t, S, X, p, A_l, A_h)`

### 2. Environment (`environment.py`)

- **OptimalExecutionEnv**: Simulation environment with regime dynamics
- **StepResult**: State transition data structure
- **Euler-Maruyama Discretization**: SDE numerical integration

### 3. Policies (`policies.py`)

- **Policy Interface**: Abstract base for all control strategies
- **Baseline Policies**: Certainty Equivalent, Naive, Oracle
- **RL Integration**: Gaussian and deterministic policy wrappers

### 4. REINFORCE Agent (`reinforce_agent.py`)

- **REINFORCEPolicy**: Policy gradient implementation
- **PolicyNetwork**: Neural network architecture (JAX/Flax)
- **Training Infrastructure**: Full training loop with evaluation

### 5. Riccati Solutions (`riccati_policies.py`)

- **RiccatiSolver**: LQR solution for linear-quadratic approximations
- **Benchmark Policies**: Optimal, Certainty Equivalent, Mean strategies
- **Analytical Solutions**: Closed-form controls where available

### 6. Comparison Tools (`comparison.py`)

- **PolicyComparator**: Performance evaluation framework
- **PaperMethodsComparator**: Three-method comparison (REINFORCE, CE, Oracle)
- **Statistical Analysis**: Monte Carlo performance assessment

## Implementation Methods

| Method | Type | Description | Key Features |
|--------|------|-------------|--------------|
| **REINFORCE** | RL | Policy gradient with Gaussian policies | JAX/Flax, Monte Carlo estimation |
| **Certainty Equivalent** | Baseline | Assumes known regime | Analytical solution |
| **Oracle** | Upper Bound | Perfect regime knowledge | Theoretical benchmark |
| **Riccati** | Analytical | LQR approximation | Closed-form solution |
| **HJB Solver** | PDE | Direct HJB equation solution | Finite difference methods |

## Configuration

### Central Parameters (`model_parameters.yaml`)

```yaml
# Physical Parameters
T: 1.0                    # Time horizon
DT: 0.01                  # Time step
SIGMA: 0.2                # Price volatility

# Impact Parameters  
RHO: 0.1                  # Temporary impact
LAMBDA_L: 0.5             # Low impact regime
LAMBDA_H: 2.0             # High impact regime
KAPPA_L: 10.0             # High resilience
KAPPA_H: 2.0              # Low resilience

# Cost Parameters
C_RUNNING: 0.01           # Running cost
C_TERMINAL: 10.0          # Terminal penalty

# Initial Conditions
Y_0: 100.0                # Initial price
X_0: 10.0                 # Initial inventory
P_0: 0.5                  # Initial belief
```

### State Space Bounds

```yaml
STATE_BOUNDS:
  t: [0.0, 1.0]          # Time
  S: [-5.0, 20.0]        # Observable state
  X: [-5.0, 20.0]        # Inventory
  p: [0.0, 1.0]          # Belief state
  alpha_l: [0.0, 5.0]    # Low regime accumulator
  alpha_h: [0.0, 5.0]    # High regime accumulator
```

## Running the Code

### Basic Usage

```python
from control_theory import (
    OptimalExecutionConfig,
    PaperMethodsComparator,
    REINFORCEConfig
)

# Setup configuration
config = OptimalExecutionConfig()
reinforce_config = REINFORCEConfig(n_episodes=1000)

# Run comparison
comparator = PaperMethodsComparator(config, reinforce_config)
results = comparator.compare_all_methods(key=random.PRNGKey(42))

# Analyze results
print(f"REINFORCE Cost: {results['reinforce'].mean_cost:.4f}")
print(f"Certainty Equivalent Cost: {results['certainty_equivalent'].mean_cost:.4f}")
print(f"Oracle Cost: {results['oracle'].mean_cost:.4f}")
```

### Individual Method Training

```python
# Train REINFORCE policy
from control_theory import train_reinforce_policy

policy, training_results = train_reinforce_policy(
    config=config,
    reinforce_config=reinforce_config,
    key=random.PRNGKey(42)
)
```

### Jupyter Analysis

```bash
# Interactive comparison and visualization
jupyter notebook control_comparison_standalone.ipynb
```

## Documentation

| File | Purpose | Content |
|------|---------|---------|
| **docs/README.md** | Project overview | Basic description and structure |
| **docs/pde_docs.md** | PDE methods | Mathematical formulation and HJB equations |
| **docs/rl_docs.md** | RL methods | Algorithm implementation details |
| **literature.md** | Research context | Comprehensive literature review |
| **CLAUDE.md** | Development guide | Code patterns and conventions |

## Mathematical Foundation

### Problem Formulation

**State Process**: 6-dimensional `(t, S_t, X_t, p_t, A^l_t, A^h_t)`

```math
\begin{aligned}
dS_t &= -[\lambda_l(u_t + \kappa_l A^l_t)p_t + \lambda_h(u_t + \kappa_h A^h_t)(1-p_t)]dt + \sigma dW_t \\
dX_t &= -u_t dt \\
dp_t &= \frac{1}{\sigma} p_t(1-p_t)[\lambda_l(u_t + \kappa_l A^l_t) - \lambda_h(u_t + \kappa_h A^h_t)]dW_t \\
dA^l_t &= (u_t + \kappa_l A^l_t)dt \\
dA^h_t &= (u_t + \kappa_h A^h_t)dt
\end{aligned}
```

**Objective**: Maximize expected profit

```math
J(u) = \mathbb{E}\left[\int_0^T \left((S_t - \rho u_t)u_t - c X_t^2\right)dt + \left(S_T X_T - C X_T^2\right)\right]
```

### Key Features

- **Regime Uncertainty**: Hidden Markov chain with two liquidity states
- **Transient Impact**: State-dependent price impact with exponential recovery
- **Belief Filtering**: Optimal inference of hidden regime from price observations
- **Finite Horizon**: Terminal time `T` with inventory liquidation objective

## API Quick Reference

### Core Classes

```python
# Configuration
config = OptimalExecutionConfig()           # Problem parameters
reinforce_config = REINFORCEConfig()        # Training parameters

# Environment
env = OptimalExecutionEnv(config)           # Simulation environment
state, reward = env.step(action)            # Environment interaction

# Policies
policy = CertaintyEquivalentPolicy(config)  # Baseline policy
policy = REINFORCEPolicy(params, config)    # Trained RL policy
action = policy.act(state, key)             # Policy evaluation

# Comparison
comparator = PaperMethodsComparator(config, reinforce_config)
results = comparator.compare_all_methods(key)  # Full comparison

# Riccati Solutions
solver = RiccatiSolver(config)              # LQR solver
policy = RiccatiOptimalPolicy(solver)       # Riccati-based policy
```

### Performance Evaluation

```python
# Single policy evaluation
result = comparator.evaluate_policy(policy, n_episodes=100, key=key)
print(f"Mean cost: {result.mean_cost}")
print(f"Std cost: {result.std_cost}")

# Statistical comparison
stats = results['reinforce'].compare_with(results['oracle'])
print(f"Relative performance: {stats['relative_cost']}")
```

## Dependencies

**Core Scientific Stack**:

- `jax[cpu]` - Numerical computation and automatic differentiation
- `flax` - Neural network framework  
- `numpy` - Array operations
- `scipy` - Scientific computing
- `pyyaml` - Configuration management

**Optional**:

- `jupyter` - Interactive analysis
- `matplotlib` - Visualization
- `pandas` - Data analysis

## Development Status

| Component | Status | Notes |
|-----------|---------|--------|
| Core Framework | ✅ Complete | Stable API with comprehensive testing |
| REINFORCE Implementation | ✅ Complete | JAX/Flax with policy gradients |
| Baseline Policies | ✅ Complete | CE, Naive, Oracle implementations |
| Riccati Solutions | ✅ Complete | LQR approximations and analytics |
| HJB Solver | 🔄 In Progress | Finite difference implementation |
| Comparison Tools | ✅ Complete | Statistical evaluation framework |
| Documentation | ✅ Complete | Comprehensive API and usage docs |

---

**Research Citation**: Cohen, S., Knochenhauer, C., & Merkel, A. (2023). Optimal adaptive control with separable drift uncertainty. arXiv preprint arXiv:2309.07091.

**License**: Academic/Research Use  
**Contact**: See paper authors for research collaboration

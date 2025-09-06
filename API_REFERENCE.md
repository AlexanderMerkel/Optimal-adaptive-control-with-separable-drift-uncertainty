# API Reference - Optimal Adaptive Control Numerics

## Core Framework Package: `control_theory`

### Import Structure

```python
from control_theory import (
    # Configuration
    OptimalExecutionConfig, default_config,
    
    # Environment 
    OptimalExecutionEnv, StepResult,
    
    # Policies
    Policy, CertaintyEquivalentPolicy, NaivePolicy, OraclePolicy, RLPolicy,
    SimpleGaussianPolicy, SimpleDeterministicPolicy,
    create_gaussian_rl_policy, create_deterministic_rl_policy,
    
    # REINFORCE Agent
    REINFORCEConfig, REINFORCEPolicy, REINFORCEAgent, PolicyNetwork,
    train_reinforce_policy,
    
    # Comparison Tools
    PolicyComparator, PolicyResult, PaperMethodsComparator,
    
    # Riccati Solutions
    RiccatiSolver, RiccatiOptimalPolicy, RiccatiCertaintyEquivalentPolicy,
    RiccatiMeanPolicy
)
```

---

## Configuration (`config.py`)

### `OptimalExecutionConfig`

**Description**: Central configuration class for optimal execution problem parameters.

```python
@dataclass
class OptimalExecutionConfig:
    # Regime Parameters
    lambda_l: float = 0.5    # Low impact regime intensity
    kappa_l: float = 10.0    # High resilience rate (liquid market)
    lambda_h: float = 2.0    # High impact regime intensity  
    kappa_h: float = 2.0     # Low resilience rate (illiquid market)
    
    # Observable Parameters
    rho: float = 0.1         # Instantaneous price impact
    sigma: float = 0.2       # Price volatility
    
    # Cost Parameters
    c: float = 0.01          # Running inventory cost
    C: float = 10.0          # Terminal inventory penalty
    
    # Problem Setup
    T: float = 1.0           # Time horizon
    Y_0: float = 100.0       # Initial asset price
    X_0: float = 10.0        # Initial inventory  
    p_0: float = 0.5         # Initial belief
    
    # Simulation Parameters
    dt: float = 0.01         # Time step
    n_steps: int = None      # Auto-computed as T/dt
```

**Properties**:

- `time_grid: jnp.ndarray` - Simulation time points
- `regime_params: tuple` - `(lambda_l, kappa_l, lambda_h, kappa_h)`
- `initial_state: jnp.ndarray` - Initial state `[Y_0, X_0, p_0, 0.0, 0.0]`

**Usage**:

```python
# Default configuration
config = OptimalExecutionConfig()

# Custom configuration
config = OptimalExecutionConfig(
    lambda_l=0.3, lambda_h=1.5,
    T=2.0, X_0=20.0
)

# Access properties
time_points = config.time_grid
initial = config.initial_state
```

### `default_config`

**Description**: Pre-configured default instance.

```python
config = default_config  # Ready-to-use default configuration
```

---

## Environment (`environment.py`)

### `StepResult`

**Description**: Named tuple containing step results.

```python
class StepResult(NamedTuple):
    next_state: jnp.ndarray    # Next state [Y, X, p, alpha_l, alpha_h]
    reward: float              # Instantaneous reward
    done: bool                 # Episode termination flag
    info: dict                 # Additional information
```

**Info Dictionary Contents**:

- `true_regime: float` - Hidden regime state (0=low, 1=high)
- `belief: float` - Updated belief probability
- `price: float` - Asset price
- `inventory: float` - Remaining inventory
- `execution_revenue: float` - Revenue from trading
- `inventory_cost: float` - Inventory holding cost
- `resilience_l/h: float` - Resilience accumulator states
- `innovation: float` - Price innovation for filtering

### `OptimalExecutionEnv`

**Description**: Main simulation environment implementing regime-switching dynamics.

**Constructor**:

```python
def __init__(self, config: OptimalExecutionConfig = default_config)
```

**Methods**:

#### `reset(key, batch_size=1) -> jnp.ndarray`

Reset environment to initial state.

```python
key = random.PRNGKey(42)
state = env.reset(key)                    # Single environment
states = env.reset(key, batch_size=10)    # Batch of 10 environments
```

**Returns**: Initial state(s) `[Y, X, p, alpha_l, alpha_h]`

#### `step(state, action, key) -> StepResult`

Take one environment step.

```python
result = env.step(state, action=2.5, key=key)
next_state = result.next_state
reward = result.reward
done = result.done
info = result.info
```

#### `batch_step(states, actions, key) -> StepResult`

Parallel batch processing.

```python
results = env.batch_step(
    states=batch_states,     # [batch_size, 5]
    actions=batch_actions,   # [batch_size]
    key=key
)
```

#### `compute_terminal_reward(state) -> float`

Terminal reward calculation: `Y*X - C*X²`

```python
terminal_value = env.compute_terminal_reward(final_state)
```

#### `generate_trajectory(policy_fn, n_steps=None, key=PRNGKey(0)) -> dict`

Generate complete trajectory with given policy.

```python
def simple_policy(state, time):
    return state[1] / (config.T - time + 1e-6)  # Linear liquidation

trajectory = env.generate_trajectory(simple_policy, key=key)

# Access trajectory data
states = trajectory['states']           # [n_steps+1, 5]
actions = trajectory['actions']         # [n_steps]
rewards = trajectory['rewards']         # [n_steps]
total_return = trajectory['total_reward']
```

**State Dynamics Implementation**:

The environment implements the mathematical model:

```python
# State: [Y, X, p, alpha_l, alpha_h]
# Y: Asset price with regime-dependent impact
# X: Inventory (decreases with trading)
# p: Belief state P(low regime | observations)
# alpha_l, alpha_h: Accumulator states for resilience

# Price evolution (hidden regime determines true impact)
dY = -true_lambda * (u + true_kappa * true_alpha) * dt + sigma * dW

# Belief update via Wonham filtering
innovation = observed_price_change - expected_price_change
dp = (1/sigma²) * p * (1-p) * [f_low - f_high] * innovation * dt

# Resilience accumulation
dalpha_l = (u + kappa_l * alpha_l) * dt
dalpha_h = (u + kappa_h * alpha_h) * dt
```

---

## Policies (`policies.py`)

### `Policy` (Abstract Base Class)

**Description**: Interface for all control policies.

```python
class Policy(ABC):
    @abstractmethod
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute control action for given state and time."""
        pass
    
    @property
    def name(self) -> str:
        """Policy name for logging/plotting."""
        return self.__class__.__name__
```

### `CertaintyEquivalentPolicy`

**Description**: Uses expected regime parameters based on current belief.

**Constructor**:

```python
policy = CertaintyEquivalentPolicy(config=config)
```

**Usage**:

```python
action = policy(state, time=0.5)
```

**Algorithm**:

- Computes expected parameters: `E[λ] = p*λ_l + (1-p)*λ_h`
- Applies deterministic control with expected values
- Adjusts trading rate based on expected price impact

### `NaivePolicy`

**Description**: Simple linear liquidation ignoring regime uncertainty.

**Constructor**:

```python
policy = NaivePolicy(config=config)
```

**Algorithm**:

- Linear liquidation: `action = inventory / remaining_time`
- No adaptation to price impact or regime changes

### `OraclePolicy`

**Description**: Perfect information policy (performance upper bound).

**Constructor & Setup**:

```python
policy = OraclePolicy(config=config)
policy.set_true_regime(regime)  # Called by environment
```

**Usage**:

```python
# Environment automatically sets regime
result = env.step(state, oracle_policy(state, time), key)
```

**Algorithm**:

- Uses true regime parameters for optimal control
- Provides theoretical performance ceiling

### `RLPolicy`

**Description**: Neural network-based policy for reinforcement learning.

**Constructor**:

```python
policy = RLPolicy(
    network=network,           # Flax neural network module
    params=network_params,     # Network parameters
    config=config,
    policy_type="gaussian"     # "gaussian" or "deterministic"
)
```

**Usage**:

```python
# Stochastic action (training)
action = policy(state, time, key=random_key)

# Deterministic action (evaluation)  
action = policy(state, time)  # Uses mean for Gaussian policies
```

**Network Types**:

- **Gaussian**: Outputs `(mean, log_std)` for stochastic policies
- **Deterministic**: Outputs single action value

### Neural Network Architectures

#### `SimpleGaussianPolicy`

```python
class SimpleGaussianPolicy(nn.Module):
    hidden_dim: int = 64
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    
    def __call__(self, x):
        # Input: [Y, X, p, alpha_l, alpha_h, time] (6 dimensions)
        # Output: (mean, log_std) for action distribution
```

#### `SimpleDeterministicPolicy`

```python
class SimpleDeterministicPolicy(nn.Module):
    hidden_dim: int = 64
    
    def __call__(self, x):
        # Input: [Y, X, p, alpha_l, alpha_h, time] (6 dimensions)  
        # Output: deterministic action
```

### Policy Factory Functions

#### `create_gaussian_rl_policy(config, hidden_dim=64, key=PRNGKey(42)) -> RLPolicy`

Create Gaussian RL policy with random initialization.

```python
policy = create_gaussian_rl_policy(
    config=config,
    hidden_dim=128,
    key=random.PRNGKey(42)
)
```

#### `create_deterministic_rl_policy(config, hidden_dim=64, key=PRNGKey(42)) -> RLPolicy`

Create deterministic RL policy.

```python
policy = create_deterministic_rl_policy(config=config)
```

---

## REINFORCE Agent (`reinforce_agent.py`)

### `REINFORCEConfig`

**Description**: Configuration for REINFORCE training.

```python
@dataclass
class REINFORCEConfig:
    n_episodes: int = 1000          # Training episodes
    hidden_dim: int = 128           # Network hidden units
    learning_rate: float = 0.001    # Adam optimizer rate
    batch_size: int = 64            # Training batch size
    log_interval: int = 100         # Logging frequency
    eval_interval: int = 1000       # Evaluation frequency  
    eval_episodes: int = 100        # Episodes for evaluation
    gamma: float = 1.0              # Discount factor (unused in finite horizon)
    entropy_coeff: float = 0.0      # Entropy regularization
```

### `PolicyNetwork`

**Description**: Neural network for REINFORCE policy.

```python
class PolicyNetwork(nn.Module):
    hidden_dim: int = 128
    
    def __call__(self, x):
        # Returns (mean, log_std) for Gaussian policy
```

### `REINFORCEPolicy`

**Description**: REINFORCE policy with training capabilities.

**Constructor**:

```python
policy = REINFORCEPolicy(
    network=PolicyNetwork(hidden_dim=128),
    config=execution_config,
    reinforce_config=reinforce_config
)
```

**Methods**:

#### `act(state, time, key) -> float`

Sample action from policy.

```python
action = policy.act(state, time, key)
```

#### `log_prob(state, time, action) -> float`

Compute log probability of action.

```python
log_p = policy.log_prob(state, time, action)
```

#### `update(trajectories, optimizer_state)`

Update policy parameters using REINFORCE.

```python
new_params, new_opt_state, metrics = policy.update(
    trajectories=batch_trajectories,
    optimizer_state=opt_state
)
```

### `REINFORCEAgent`

**Description**: Full REINFORCE training agent.

**Constructor**:

```python
agent = REINFORCEAgent(
    env=environment,
    config=execution_config,
    reinforce_config=reinforce_config,
    key=random.PRNGKey(42)
)
```

**Methods**:

#### `train() -> dict`

Run complete training loop.

```python
results = agent.train()

# Access training metrics
final_policy = results['policy']
training_curves = results['training_metrics']
evaluation_results = results['evaluation_results']
```

#### `evaluate(n_episodes=100) -> dict`

Evaluate current policy.

```python
eval_results = agent.evaluate(n_episodes=50)
mean_return = eval_results['mean_return']
std_return = eval_results['std_return']
```

### `train_reinforce_policy(config, reinforce_config, key) -> tuple`

**Description**: High-level training function.

```python
policy, results = train_reinforce_policy(
    config=OptimalExecutionConfig(),
    reinforce_config=REINFORCEConfig(n_episodes=500),
    key=random.PRNGKey(42)
)

# Use trained policy
action = policy.act(state, time, key)
```

**Returns**: `(trained_policy, training_results)`

---

## Comparison Tools (`comparison.py`)

### `PolicyResult`

**Description**: Results from policy evaluation.

```python
@dataclass
class PolicyResult:
    policy_name: str
    mean_cost: float          # Average total cost
    std_cost: float           # Standard deviation
    mean_return: float        # Average total return (negative cost)
    std_return: float
    trajectories: List[dict]  # Individual trajectory data
    execution_times: List[float]  # Time to complete liquidation
    final_inventories: List[float]  # Remaining inventory at end
```

**Methods**:

#### `compare_with(other: PolicyResult) -> dict`

Statistical comparison between policies.

```python
stats = result1.compare_with(result2)

print(f"Relative cost: {stats['relative_cost']:.3f}")
print(f"Improvement: {stats['cost_improvement']:.1%}")  
print(f"T-statistic: {stats['t_statistic']:.2f}")
print(f"P-value: {stats['p_value']:.4f}")
```

### `PolicyComparator`

**Description**: Framework for comparing multiple policies.

**Constructor**:

```python
comparator = PolicyComparator(
    env=environment,
    config=config,
    n_evaluation_episodes=100
)
```

**Methods**:

#### `evaluate_policy(policy, n_episodes=None, key=PRNGKey(0)) -> PolicyResult`

Evaluate single policy performance.

```python
result = comparator.evaluate_policy(
    policy=certainty_equivalent_policy,
    n_episodes=200,
    key=key
)
```

#### `compare_policies(policies: dict, key=PRNGKey(0)) -> dict`

Compare multiple policies.

```python
policies = {
    'CE': CertaintyEquivalentPolicy(config),
    'Naive': NaivePolicy(config),
    'Oracle': OraclePolicy(config)
}

results = comparator.compare_policies(policies, key=key)

# Access results
for name, result in results.items():
    print(f"{name}: {result.mean_cost:.3f} ± {result.std_cost:.3f}")
```

### `PaperMethodsComparator`

**Description**: Specialized comparator for paper's three methods.

**Constructor**:

```python
comparator = PaperMethodsComparator(
    config=execution_config,
    reinforce_config=reinforce_config
)
```

**Methods**:

#### `compare_all_methods(key=PRNGKey(0)) -> dict`

Compare REINFORCE, Certainty Equivalent, and Oracle policies.

```python
results = comparator.compare_all_methods(key=key)

reinforce_result = results['reinforce']
ce_result = results['certainty_equivalent'] 
oracle_result = results['oracle']

# Performance comparison
reinforce_vs_oracle = reinforce_result.compare_with(oracle_result)
print(f"REINFORCE achieves {reinforce_vs_oracle['relative_performance']:.1%} of Oracle performance")
```

#### `train_and_evaluate_reinforce(key=PRNGKey(0)) -> tuple`

Train REINFORCE policy and return policy + results.

```python
policy, result = comparator.train_and_evaluate_reinforce(key=key)
```

---

## Riccati Solutions (`riccati_policies.py`)

### `RiccatiSolver`

**Description**: Linear-quadratic approximation solver.

**Constructor**:

```python
solver = RiccatiSolver(config=config)
```

**Methods**:

#### `solve_riccati_equation() -> jnp.ndarray`

Solve finite-horizon Riccati equation.

```python
P = solver.solve_riccati_equation()  # [n_steps+1, state_dim, state_dim]
```

#### `compute_optimal_gain(time_index: int) -> jnp.ndarray`

Get optimal feedback gain matrix.

```python
K = solver.compute_optimal_gain(time_index=50)
```

### Riccati-Based Policies

#### `RiccatiOptimalPolicy`

Optimal LQR policy using full state feedback.

```python
policy = RiccatiOptimalPolicy(solver)
action = policy(state, time)
```

#### `RiccatiCertaintyEquivalentPolicy`

LQR with certainty equivalent approximation.

```python
policy = RiccatiCertaintyEquivalentPolicy(solver)
```

#### `RiccatiMeanPolicy`

LQR using mean regime parameters.

```python
policy = RiccatiMeanPolicy(solver)
```

---

## Complete Usage Examples

### Basic Policy Comparison

```python
import jax.numpy as jnp
from jax import random
from control_theory import *

# Setup
config = OptimalExecutionConfig(T=1.0, X_0=10.0)
env = OptimalExecutionEnv(config)
key = random.PRNGKey(42)

# Create policies
policies = {
    'Certainty Equivalent': CertaintyEquivalentPolicy(config),
    'Naive': NaivePolicy(config), 
    'Oracle': OraclePolicy(config)
}

# Compare performance
comparator = PolicyComparator(env, config, n_evaluation_episodes=100)
results = comparator.compare_policies(policies, key=key)

# Print results
for name, result in results.items():
    print(f"{name}: Cost = {result.mean_cost:.3f} ± {result.std_cost:.3f}")
```

### REINFORCE Training

```python
# Training configuration
reinforce_config = REINFORCEConfig(
    n_episodes=2000,
    hidden_dim=128, 
    learning_rate=0.001,
    batch_size=64
)

# Train policy
key = random.PRNGKey(42)
policy, training_results = train_reinforce_policy(
    config=config,
    reinforce_config=reinforce_config,
    key=key
)

# Evaluate trained policy
comparator = PolicyComparator(env, config)
result = comparator.evaluate_policy(policy, n_episodes=200, key=key)
print(f"REINFORCE Policy: Cost = {result.mean_cost:.3f}")
```

### Complete Paper Comparison

```python
# Setup for paper comparison
config = OptimalExecutionConfig()
reinforce_config = REINFORCEConfig(n_episodes=1000)

# Run all methods
comparator = PaperMethodsComparator(config, reinforce_config)
results = comparator.compare_all_methods(key=random.PRNGKey(42))

# Analysis
reinforce_cost = results['reinforce'].mean_cost
oracle_cost = results['oracle'].mean_cost
efficiency = reinforce_cost / oracle_cost

print(f"REINFORCE Cost: {reinforce_cost:.4f}")
print(f"Oracle Cost: {oracle_cost:.4f}")
print(f"Efficiency: {efficiency:.1%}")
```

### Trajectory Analysis

```python
# Generate detailed trajectory
policy = CertaintyEquivalentPolicy(config)
trajectory = env.generate_trajectory(policy, key=key)

# Extract data
states = trajectory['states']           # [n_steps+1, 5]
prices = states[:, 0]                   # Asset prices
inventories = states[:, 1]              # Inventory levels  
beliefs = states[:, 2]                  # Belief evolution
actions = trajectory['actions']         # Trading rates
rewards = trajectory['rewards']         # Instantaneous rewards

# Analysis
total_return = trajectory['total_reward']
final_inventory = inventories[-1]
avg_belief = jnp.mean(beliefs)

print(f"Total Return: {total_return:.3f}")
print(f"Final Inventory: {final_inventory:.3f}")
print(f"Average Belief: {avg_belief:.3f}")
```

---

## Error Handling & Validation

### Common Issues

1. **Invalid State Bounds**: Belief `p` automatically clipped to `[0,1]`
2. **Negative Actions**: Policies automatically enforce non-negative trading
3. **Inventory Constraints**: Actions limited by `inventory/dt` to prevent over-selling
4. **Time Bounds**: Remaining time clamped to avoid division by zero

### Configuration Validation

The `OptimalExecutionConfig` validates parameters in `__post_init__`:

- Impact parameters: `lambda_l < lambda_h`, both non-negative  
- Resilience: `kappa_l > kappa_h`, both positive
- Belief: `0 <= p_0 <= 1`
- Costs: `c, C > 0`
- Time: `T, dt > 0`

### JIT Compilation

Most computationally intensive functions are JIT-compiled:

- Environment step functions
- Policy action computations  
- Neural network forward passes
- Training updates

This provides significant performance improvements but requires JAX-compatible code.

---

*This API reference covers the complete control_theory framework. For mathematical foundations, see `docs/pde_docs.md` and `docs/rl_docs.md`. For usage examples, see `control_comparison_standalone.ipynb`.*

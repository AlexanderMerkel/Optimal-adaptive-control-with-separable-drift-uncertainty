"""
Control Policy Implementations

This module provides concrete implementations of the ControlPolicy interface
for different types of control strategies commonly used in stochastic control.

Policy Types:
    - RiccatiPolicy: LQR-based policies using Riccati equation solutions
    - NeuralPolicy: Neural network-based policies for reinforcement learning
    - NetworkArchitectureRegistry: Centralized management of network architectures
"""

from typing import Optional, Dict, Any, Callable, Union
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random

from .core import State, Action, ControlPolicy, create_action
from ..utils import RiccatiSolver, Config


class RiccatiPolicy(ControlPolicy):
    """
    Control policy based on Riccati equation solutions for LQR problems.
    
    Implements optimal linear feedback control: u(t) = -K(t) * x(t)
    where K(t) is the solution to the differential Riccati equation.
    
    This policy is suitable for:
    - Linear-quadratic control problems
    - Certainty equivalent control
    - Mean-field control with averaged parameters
    - Robust control with worst-case parameters
    """
    
    def __init__(self, 
                 riccati_solver: RiccatiSolver,
                 lambda_func: Union[float, Callable[[State, float], float]],
                 rho: float,
                 state_indices: Optional[Dict[str, int]] = None):
        """
        Initialize Riccati-based policy.
        
        Args:
            riccati_solver: Solver for Riccati equations
            lambda_func: Lambda parameter (float) or function of state/time
            rho: Temporary impact parameter 
            state_indices: Mapping of state variable names to indices
        """
        self.riccati_solver = riccati_solver
        self.lambda_func = lambda_func
        self.rho = rho
        self.state_indices = state_indices or {'X': 2}  # Default inventory index
        
        # Pre-solve Riccati equation if lambda is constant
        if isinstance(lambda_func, (int, float)):
            self.K_trajectory = riccati_solver.solve(float(lambda_func))
            self._constant_lambda = True
        else:
            self._constant_lambda = False
            # For variable lambda, we'll solve on-demand or use grid interpolation
            self._setup_lambda_grid()
    
    def _setup_lambda_grid(self):
        """Setup grid-based interpolation for variable lambda functions."""
        # Create reasonable lambda range for interpolation
        lambda_min, lambda_max = 0.1, 3.0  # Reasonable range for most problems
        self.lambda_grid = jnp.linspace(lambda_min, lambda_max, 101)
        self.K_solutions = self.riccati_solver.solve_grid(self.lambda_grid)
    
    def compute_action(self, state: State, time: Optional[float] = None, 
                      key: Optional[random.PRNGKey] = None) -> Action:
        """
        Compute optimal linear feedback control action.
        
        Args:
            state: Current state
            time: Current time (for time step indexing)
            key: Unused (deterministic policy)
            
        Returns:
            Optimal control action
        """
        # Extract inventory position (or relevant state variable)
        if isinstance(self.state_indices['X'], int):
            X = state.data[self.state_indices['X']]
        else:
            # Handle batch case
            X = state.data[:, self.state_indices['X']]
        
        # Get Riccati gain K(t)
        if self._constant_lambda:
            # Use pre-computed trajectory
            if time is not None:
                # Convert time to index using JAX-compatible operations
                time_idx = jnp.floor(time * len(self.K_trajectory) / self.riccati_solver.config.T).astype(int)
                time_idx = jnp.clip(time_idx, 0, len(self.K_trajectory) - 1)
            else:
                time_idx = 0
            K_current = self.K_trajectory[time_idx]
        else:
            # Variable lambda case - evaluate lambda function
            lambda_val = self.lambda_func(state, time or 0.0)
            if time is not None:
                time_idx = jnp.floor(time * len(self.K_solutions) / self.riccati_solver.config.T).astype(int)
                time_idx = jnp.clip(time_idx, 0, len(self.K_solutions) - 1)
            else:
                time_idx = 0
            K_current = self.riccati_solver.interpolate_solution(
                self.K_solutions, self.lambda_grid, lambda_val, time_idx
            )
        
        # Compute optimal action: u = -(lambda / rho) * K(t) * X(t)
        if self._constant_lambda:
            lambda_val = self.lambda_func if isinstance(self.lambda_func, (int, float)) else 1.0
        else:
            lambda_val = self.lambda_func(state, time or 0.0)
            
        action_value = -(lambda_val / self.rho) * K_current * X
        
        return create_action(
            data=jnp.atleast_1d(action_value),
            policy_type="riccati",
            lambda_value=lambda_val,
            riccati_gain=K_current
        )
    
    @property
    def is_stochastic(self) -> bool:
        """Riccati policies are deterministic."""
        return False
        
    @property 
    def is_time_varying(self) -> bool:
        """Riccati policies are typically time-varying in finite horizon."""
        return True


class NeuralPolicy(ControlPolicy):
    """
    Neural network-based control policy for reinforcement learning.
    
    Supports both deterministic and stochastic policies through different
    network architectures and output processing.
    """
    
    def __init__(self,
                 network: nn.Module,
                 params: Dict[str, Any],
                 policy_type: str = "gaussian",
                 action_bounds: Optional[tuple] = None):
        """
        Initialize neural network policy.
        
        Args:
            network: Flax neural network module
            params: Network parameters
            policy_type: Type of policy ("gaussian", "deterministic", "discrete")
            action_bounds: Optional bounds for action clipping
        """
        self.network = network
        self.params = params
        self.policy_type = policy_type
        self.action_bounds = action_bounds
        
    def compute_action(self, state: State, time: Optional[float] = None,
                      key: Optional[random.PRNGKey] = None) -> Action:
        """
        Compute action using neural network.
        
        Args:
            state: Current state
            time: Current time (may be unused)
            key: Random key for stochastic policies
            
        Returns:
            Neural network action
        """
        # Forward pass through network
        if self.policy_type == "gaussian":
            mean, log_std = self.network.apply({"params": self.params}, state.data)
            mean, log_std = mean.squeeze(), log_std.squeeze()
            
            if key is not None:
                # Sample from Gaussian distribution
                std = jnp.exp(log_std)
                action_raw = mean + std * random.normal(key, mean.shape)
            else:
                # Use mean for deterministic evaluation
                action_raw = mean
                
            # Apply tanh squashing if bounds are specified
            if self.action_bounds is not None:
                action_value = jnp.tanh(action_raw) * self.action_bounds[1]
            else:
                action_value = action_raw
                
            return create_action(
                data=jnp.atleast_1d(action_value),
                policy_type="neural_gaussian",
                mean=mean,
                log_std=log_std,
                raw_action=action_raw
            )
            
        elif self.policy_type == "deterministic":
            action_raw = self.network.apply({"params": self.params}, state.data)
            
            # Apply bounds if specified
            if self.action_bounds is not None:
                action_value = jnp.tanh(action_raw) * self.action_bounds[1]
            else:
                action_value = action_raw
                
            return create_action(
                data=jnp.atleast_1d(action_value),
                policy_type="neural_deterministic"
            )
            
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}")
    
    @property
    def is_stochastic(self) -> bool:
        """Neural policies can be stochastic depending on type."""
        return self.policy_type in ["gaussian", "discrete"]
        
    @property
    def is_time_varying(self) -> bool:
        """Neural policies can be time-varying if time is an input."""
        return False  # Default assumption, can be overridden


class NetworkArchitectureRegistry:
    """
    Centralized registry for neural network architectures.
    
    Provides a clean interface for creating and managing different network
    architectures used across the control theory framework.
    """
    
    _architectures = {}
    
    @classmethod
    def register(cls, name: str, network_class: type, default_config: Dict[str, Any]):
        """
        Register a network architecture.
        
        Args:
            name: Architecture name
            network_class: Network class (Flax module)
            default_config: Default configuration parameters
        """
        cls._architectures[name] = {
            'class': network_class,
            'default_config': default_config
        }
    
    @classmethod
    def create_network(cls, name: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Create network instance from registry.
        
        Args:
            name: Registered architecture name
            config: Override configuration parameters
            
        Returns:
            Network instance
        """
        if name not in cls._architectures:
            raise ValueError(f"Unknown architecture: {name}")
            
        arch_info = cls._architectures[name]
        network_config = arch_info['default_config'].copy()
        if config:
            network_config.update(config)
            
        return arch_info['class'](**network_config)
    
    @classmethod
    def list_architectures(cls) -> list:
        """List all registered architectures."""
        return list(cls._architectures.keys())


# Standard network architectures
class GaussianPolicyNetwork(nn.Module):
    """Standard Gaussian policy network for continuous control."""
    
    hidden_dim: int = 64
    action_dim: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


class DeterministicPolicyNetwork(nn.Module):
    """Deterministic policy network for continuous control."""
    
    hidden_dim: int = 64
    action_dim: int = 1
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        
        action = nn.Dense(self.action_dim)(x)
        return action


class ValueNetwork(nn.Module):
    """Value function network for critic-based methods."""
    
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        
        value = nn.Dense(1)(x)
        return value


# Register standard architectures
NetworkArchitectureRegistry.register(
    "gaussian_policy",
    GaussianPolicyNetwork,
    {"hidden_dim": 64, "action_dim": 1}
)

NetworkArchitectureRegistry.register(
    "deterministic_policy", 
    DeterministicPolicyNetwork,
    {"hidden_dim": 64, "action_dim": 1}
)

NetworkArchitectureRegistry.register(
    "value_function",
    ValueNetwork,
    {"hidden_dim": 64}
)
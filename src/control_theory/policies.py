"""
Control Policies for Optimal Execution with Regime Uncertainty

Implements different control strategies for comparison:
1. CertaintyEquivalentPolicy: Uses expected parameters E[λ], E[κ]
2. NaivePolicy: Simple linear liquidation
3. OraclePolicy: Knows true regime (upper bound)  
4. RLPolicy: Interface for reinforcement learning policies
"""

import jax
import jax.numpy as jnp
from jax import random
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import flax.linen as nn

from .config import OptimalExecutionConfig, default_config


class Policy(ABC):
    """Base class for control policies."""
    
    @abstractmethod
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """
        Compute control action for given state and time.
        
        Args:
            state: Current state [Y, X, p, alpha_l, alpha_h]
            time: Current time
            
        Returns:
            Control action (trading rate)
        """
        pass
    
    @property
    def name(self) -> str:
        """Policy name for logging/plotting."""
        return self.__class__.__name__


class CertaintyEquivalentPolicy(Policy):
    """
    Certainty Equivalent control policy.
    
    Uses expected regime parameters based on current belief:
    E[λ] = p * λ_l + (1-p) * λ_h
    E[κ] = p * κ_l + (1-p) * κ_h
    
    Then applies deterministic control as if these were the true parameters.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize CE policy with configuration."""
        self.config = config
        
        # JIT compile for performance
        self._compute_action = jax.jit(self._compute_action_impl)
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute CE control action."""
        return self._compute_action(state, time)
    
    def _compute_action_impl(self, state: jnp.ndarray, time: float) -> float:
        """JIT-compiled CE action computation."""
        Y, X, p, alpha_l, alpha_h = state
        
        # Expected parameters based on belief
        lambda_l, kappa_l, lambda_h, kappa_h = self.config.regime_params
        expected_lambda = p * lambda_l + (1 - p) * lambda_h
        expected_kappa = p * kappa_l + (1 - p) * kappa_h  
        expected_alpha = p * alpha_l + (1 - p) * alpha_h
        
        # Remaining time
        remaining_time = self.config.T - time
        remaining_time = jnp.maximum(remaining_time, 1e-6)  # Avoid division by zero
        
        # Simple CE heuristic: liquidate inventory over remaining time
        # with adjustment for expected impact
        base_rate = X / remaining_time
        
        # Adjust for price impact (reduce rate if high impact expected)
        impact_adjustment = 1.0 / (1.0 + expected_lambda * expected_alpha / self.config.rho)
        
        action = base_rate * impact_adjustment
        
        # Ensure action doesn't exceed inventory
        action = jnp.minimum(action, X / self.config.dt)
        action = jnp.maximum(action, 0.0)  # No buying
        
        return action
    
    @property 
    def name(self) -> str:
        return "Certainty Equivalent"


class NaivePolicy(Policy):
    """
    Naive linear liquidation policy.
    
    Simply liquidates inventory linearly over time, ignoring regime uncertainty
    and price impact effects.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize naive policy."""
        self.config = config
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute naive linear liquidation action."""
        Y, X, p, alpha_l, alpha_h = state
        
        # Remaining time
        remaining_time = self.config.T - time
        remaining_time = jnp.maximum(remaining_time, 1e-6)
        
        # Linear liquidation rate
        action = X / remaining_time
        
        # Ensure non-negative and feasible
        action = jnp.maximum(action, 0.0)
        action = jnp.minimum(action, X / self.config.dt)
        
        return action
    
    @property
    def name(self) -> str:
        return "Naive Linear"


class OraclePolicy(Policy):
    """
    Oracle policy that knows the true regime.
    
    This provides an upper bound on performance since it has perfect information
    about the hidden regime state.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize oracle policy."""
        self.config = config
        self.true_regime = None  # Set by environment
        
        # JIT compile
        self._compute_action = jax.jit(self._compute_action_impl)
    
    def set_true_regime(self, regime: float):
        """Set the true regime (called by environment)."""
        self.true_regime = regime
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute oracle control action."""
        if self.true_regime is None:
            # Fallback to CE if regime not set
            return CertaintyEquivalentPolicy(self.config)(state, time)
        
        return self._compute_action(state, time, self.true_regime)
    
    def _compute_action_impl(self, state: jnp.ndarray, time: float, true_regime: float) -> float:
        """JIT-compiled oracle action computation."""
        Y, X, p, alpha_l, alpha_h = state
        
        # True parameters
        lambda_l, kappa_l, lambda_h, kappa_h = self.config.regime_params
        true_lambda = (1 - true_regime) * lambda_l + true_regime * lambda_h
        true_kappa = (1 - true_regime) * kappa_l + true_regime * kappa_h
        true_alpha = (1 - true_regime) * alpha_l + true_regime * alpha_h
        
        # Remaining time
        remaining_time = self.config.T - time
        remaining_time = jnp.maximum(remaining_time, 1e-6)
        
        # Optimal liquidation with known parameters
        base_rate = X / remaining_time
        
        # Adjust for true impact
        impact_adjustment = 1.0 / (1.0 + true_lambda * true_alpha / self.config.rho)
        
        action = base_rate * impact_adjustment
        
        # Constraints
        action = jnp.minimum(action, X / self.config.dt)
        action = jnp.maximum(action, 0.0)
        
        return action
    
    @property
    def name(self) -> str:
        return "Oracle"


class RLPolicy(Policy):
    """
    Reinforcement Learning policy interface.
    
    Can be used with different RL algorithms (REINFORCE, PPO, etc.).
    Uses a neural network to map states to actions.
    """
    
    def __init__(self, 
                 network: nn.Module,
                 params: Dict[str, Any],
                 config: OptimalExecutionConfig = default_config,
                 policy_type: str = "gaussian"):
        """
        Initialize RL policy.
        
        Args:
            network: Neural network (Flax module)
            params: Network parameters
            config: Problem configuration
            policy_type: "gaussian" or "deterministic"
        """
        self.network = network
        self.params = params
        self.config = config
        self.policy_type = policy_type
        
        # JIT compile network forward pass
        self._forward = jax.jit(self.network.apply)
    
    def __call__(self, state: jnp.ndarray, time: float, key: Optional[random.PRNGKey] = None) -> float:
        """Compute RL policy action."""
        # Add time to state if network expects it
        network_input = jnp.concatenate([state, jnp.array([time])])
        
        if self.policy_type == "gaussian":
            mean, log_std = self._forward({"params": self.params}, network_input)
            mean, log_std = mean.squeeze(), log_std.squeeze()
            
            if key is not None:
                # Sample from policy
                std = jnp.exp(log_std)
                action = mean + std * random.normal(key)
            else:
                # Use mean for deterministic evaluation
                action = mean
        
        elif self.policy_type == "deterministic":
            action = self._forward({"params": self.params}, network_input).squeeze()
        
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        
        # Apply constraints (non-negative, inventory bound)
        Y, X, p, alpha_l, alpha_h = state
        action = jnp.maximum(action, 0.0)
        action = jnp.minimum(action, X / self.config.dt)
        
        return action
    
    @property
    def name(self) -> str:
        return f"RL-{self.policy_type.capitalize()}"


# Simple neural network architectures for RL policies
class SimpleGaussianPolicy(nn.Module):
    """Simple feedforward network for Gaussian policies."""
    
    hidden_dim: int = 64
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    
    @nn.compact
    def __call__(self, x):
        # Two hidden layers
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        
        # Output mean and log_std
        mean = nn.Dense(1)(x)
        log_std = nn.Dense(1)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


class SimpleDeterministicPolicy(nn.Module):
    """Simple feedforward network for deterministic policies."""
    
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        # Two hidden layers  
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        
        # Single action output
        action = nn.Dense(1)(x)
        return action


# Utility functions for creating RL policies
def create_gaussian_rl_policy(config: OptimalExecutionConfig = default_config,
                            hidden_dim: int = 64,
                            key: random.PRNGKey = random.PRNGKey(42)) -> RLPolicy:
    """Create RL policy with Gaussian action distribution."""
    network = SimpleGaussianPolicy(hidden_dim=hidden_dim)
    
    # Initialize parameters
    dummy_input = jnp.ones(6)  # State dim (5) + time (1)
    params = network.init(key, dummy_input)
    
    return RLPolicy(network, params, config, "gaussian")


def create_deterministic_rl_policy(config: OptimalExecutionConfig = default_config,
                                  hidden_dim: int = 64, 
                                  key: random.PRNGKey = random.PRNGKey(42)) -> RLPolicy:
    """Create RL policy with deterministic actions.""" 
    network = SimpleDeterministicPolicy(hidden_dim=hidden_dim)
    
    # Initialize parameters
    dummy_input = jnp.ones(6)  # State dim (5) + time (1)
    params = network.init(key, dummy_input)
    
    return RLPolicy(network, params, config, "deterministic")
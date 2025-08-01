"""
Configuration for Optimal Execution with Regime Uncertainty

Parameters from "Optimal adaptive control with separable drift uncertainty"
All parameters are problem-specific for optimal execution.
"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class OptimalExecutionConfig:
    """Configuration for optimal execution with regime uncertainty problem."""
    
    # ========== Regime Parameters ==========
    # Low impact regime (liquid market)
    lambda_l: float = 0.5    # Low price impact intensity
    kappa_l: float = 10.0    # High resilience rate (fast reversion)
    
    # High impact regime (illiquid market)  
    lambda_h: float = 2.0    # High price impact intensity
    kappa_h: float = 2.0     # Low resilience rate (slow reversion)
    
    # Observable parameters
    rho: float = 0.1         # Instantaneous price impact (observable)
    sigma: float = 0.2       # Price volatility
    
    # ========== Cost Parameters ==========
    c: float = 0.01          # Running inventory cost
    C: float = 10.0          # Terminal inventory penalty
    
    # ========== Problem Setup ==========
    T: float = 1.0           # Time horizon
    
    # Initial conditions
    Y_0: float = 100.0       # Initial asset price
    X_0: float = 10.0        # Initial inventory
    p_0: float = 0.5         # Initial belief (P(low regime))
    
    # ========== Simulation Parameters ==========
    dt: float = 0.01         # Time step
    n_steps: int = None      # Number of steps (computed from T/dt)
    
    def __post_init__(self):
        """Compute derived parameters."""
        if self.n_steps is None:
            self.n_steps = int(self.T / self.dt)
        
        # Validate parameters
        assert self.lambda_l >= 0 and self.lambda_h >= 0, "Impact parameters must be non-negative"
        assert self.kappa_l > 0 and self.kappa_h > 0, "Resilience rates must be positive"
        assert self.lambda_l < self.lambda_h, "Low regime should have lower impact"
        assert self.kappa_l > self.kappa_h, "Low regime should have higher resilience"
        assert 0 <= self.p_0 <= 1, "Initial belief must be a probability"
        assert self.c > 0 and self.C > 0, "Cost parameters must be positive"
        assert self.T > 0 and self.dt > 0, "Time parameters must be positive"
    
    @property
    def time_grid(self) -> jnp.ndarray:
        """Time grid for simulation."""
        return jnp.linspace(0, self.T, self.n_steps + 1)
    
    @property
    def regime_params(self) -> tuple:
        """Regime parameters as (lambda_l, kappa_l, lambda_h, kappa_h)."""
        return (self.lambda_l, self.kappa_l, self.lambda_h, self.kappa_h)
    
    @property
    def initial_state(self) -> jnp.ndarray:
        """Initial state [Y, X, p, alpha_l, alpha_h]."""
        return jnp.array([self.Y_0, self.X_0, self.p_0, 0.0, 0.0])


# Default configuration instance
default_config = OptimalExecutionConfig()
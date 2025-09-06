"""
Riccati-based Control Policies for Optimal Execution

Implements control policies based on the Riccati equation solution from:
"Optimal adaptive control with separable drift uncertainty" 

All policies use the linear feedback control law:
u*(s) = v₀(T-s) * (v₁(T-s)*X(s) + v₂(T-s)*α(s))

where v₀, v₁, v₂ are coefficient functions with closed-form solutions.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from .policies import Policy
from .config import OptimalExecutionConfig, default_config


class RiccatiSolver:
    """
    Solves the Riccati equations for optimal control coefficients.
    
    Implements the closed-form solutions from Proposition 1 of the paper:
    - v₀(τ) = 1/(1 + τ/(2ρ))
    - v₁(τ) = (C/ρ)/(1 + τ/(2ρ))²
    - v₂(τ) = (2κρ/(1 + τ/(2ρ))) * (1 - 1/(1 + τ/(2ρ)))
    
    where τ = T - t is time-to-maturity.
    """
    
    def __init__(self, rho: float, C: float, kappa: float):
        """
        Initialize Riccati solver with problem parameters.
        
        Args:
            rho: Instantaneous price impact parameter
            C: Terminal inventory penalty
            kappa: Resilience rate
        """
        self.rho = rho
        self.C = C
        self.kappa = kappa
        
        # JIT compile coefficient functions
        self._v0 = jax.jit(self._v0_impl)
        self._v1 = jax.jit(self._v1_impl)
        self._v2 = jax.jit(self._v2_impl)
    
    def _v0_impl(self, tau: float) -> float:
        """Coefficient v₀(τ) = 1/(1 + τ/(2ρ))"""
        return 1.0 / (1.0 + tau / (2.0 * self.rho))
    
    def _v1_impl(self, tau: float) -> float:
        """Coefficient v₁(τ) = (C/ρ)/(1 + τ/(2ρ))²"""
        denominator = 1.0 + tau / (2.0 * self.rho)
        return (self.C / self.rho) / (denominator ** 2)
    
    def _v2_impl(self, tau: float) -> float:
        """Coefficient v₂(τ) = (2κρ/(1 + τ/(2ρ))) * (1 - 1/(1 + τ/(2ρ)))"""
        factor = 1.0 / (1.0 + tau / (2.0 * self.rho))
        return 2.0 * self.kappa * self.rho * factor * (1.0 - factor)
    
    def get_coefficients(self, tau: float) -> Tuple[float, float, float]:
        """
        Get all three coefficients for given time-to-maturity.
        
        Args:
            tau: Time-to-maturity (T - t)
            
        Returns:
            Tuple of (v₀, v₁, v₂)
        """
        return self._v0(tau), self._v1(tau), self._v2(tau)
    
    def compute_control(self, tau: float, X: float, alpha: float) -> float:
        """
        Compute optimal control using Riccati feedback law.
        
        Args:
            tau: Time-to-maturity (T - t)
            X: Current inventory
            alpha: Current transient impact accumulator
            
        Returns:
            Optimal control u*
        """
        v0, v1, v2 = self.get_coefficients(tau)
        return v0 * (v1 * X + v2 * alpha)


class RiccatiOptimalPolicy(Policy):
    """
    Riccati-based optimal policy with known parameters.
    
    This is analogous to the Oracle policy but uses the proper
    Riccati equation solution instead of heuristics.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize Riccati optimal policy."""
        self.config = config
        self.true_regime = None  # Set by environment for oracle-like behavior
        
        # Create solvers for both regimes
        self.solver_l = RiccatiSolver(config.rho, config.C, config.kappa_l)
        self.solver_h = RiccatiSolver(config.rho, config.C, config.kappa_h)
        
        # JIT compile
        self._compute_action = jax.jit(self._compute_action_impl)
    
    def set_true_regime(self, regime: float):
        """Set the true regime (0 = low, 1 = high)."""
        self.true_regime = regime
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute Riccati optimal control action."""
        if self.true_regime is None:
            # Default to low regime if not set
            self.true_regime = 0.0
        
        return self._compute_action(state, time, self.true_regime)
    
    def _compute_action_impl(self, state: jnp.ndarray, time: float, true_regime: float) -> float:
        """JIT-compiled Riccati optimal action computation."""
        Y, X, p, alpha_l, alpha_h = state
        
        # Time-to-maturity
        tau = self.config.T - time
        tau = jnp.maximum(tau, 1e-6)  # Avoid division by zero
        
        # Select true alpha based on regime
        true_alpha = (1 - true_regime) * alpha_l + true_regime * alpha_h
        
        # Use appropriate solver based on regime
        # Note: JAX requires both branches to be computed for JIT
        control_l = self.solver_l.compute_control(tau, X, true_alpha)
        control_h = self.solver_h.compute_control(tau, X, true_alpha)
        
        # Select based on regime
        action = (1 - true_regime) * control_l + true_regime * control_h
        
        # Apply constraints
        action = jnp.maximum(action, 0.0)
        action = jnp.minimum(action, X / self.config.dt)
        
        return action
    
    @property
    def name(self) -> str:
        return "Riccati Optimal"


class RiccatiCertaintyEquivalentPolicy(Policy):
    """
    Riccati-based Certainty Equivalent policy.
    
    Uses expected parameters E[λ], E[κ] based on current belief p
    in the Riccati framework.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize Riccati CE policy."""
        self.config = config
        
        # JIT compile
        self._compute_action = jax.jit(self._compute_action_impl)
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute Riccati CE control action."""
        return self._compute_action(state, time)
    
    def _compute_action_impl(self, state: jnp.ndarray, time: float) -> float:
        """JIT-compiled Riccati CE action computation."""
        Y, X, p, alpha_l, alpha_h = state
        
        # Expected parameters based on belief
        expected_kappa = p * self.config.kappa_l + (1 - p) * self.config.kappa_h
        expected_alpha = p * alpha_l + (1 - p) * alpha_h
        
        # Time-to-maturity
        tau = self.config.T - time
        tau = jnp.maximum(tau, 1e-6)
        
        # Create solver with expected kappa
        # Note: For JIT compatibility, we compute coefficients directly
        v0 = 1.0 / (1.0 + tau / (2.0 * self.config.rho))
        v1 = (self.config.C / self.config.rho) / ((1.0 + tau / (2.0 * self.config.rho)) ** 2)
        v2 = 2.0 * expected_kappa * self.config.rho * v0 * (1.0 - v0)
        
        # Compute control
        action = v0 * (v1 * X + v2 * expected_alpha)
        
        # Apply constraints
        action = jnp.maximum(action, 0.0)
        action = jnp.minimum(action, X / self.config.dt)
        
        return action
    
    @property
    def name(self) -> str:
        return "Riccati CE"


class RiccatiMeanPolicy(Policy):
    """
    Riccati-based Mean policy.
    
    Computes the mean of regime-specific Riccati solutions,
    weighted by the current belief.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize Riccati Mean policy."""
        self.config = config
        
        # Create solvers for both regimes
        self.solver_l = RiccatiSolver(config.rho, config.C, config.kappa_l)
        self.solver_h = RiccatiSolver(config.rho, config.C, config.kappa_h)
        
        # JIT compile
        self._compute_action = jax.jit(self._compute_action_impl)
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute Riccati Mean control action."""
        return self._compute_action(state, time)
    
    def _compute_action_impl(self, state: jnp.ndarray, time: float) -> float:
        """JIT-compiled Riccati Mean action computation."""
        Y, X, p, alpha_l, alpha_h = state
        
        # Time-to-maturity
        tau = self.config.T - time
        tau = jnp.maximum(tau, 1e-6)
        
        # Compute control for each regime with its own alpha
        control_l = self.solver_l.compute_control(tau, X, alpha_l)
        control_h = self.solver_h.compute_control(tau, X, alpha_h)
        
        # Weighted average based on belief
        action = p * control_l + (1 - p) * control_h
        
        # Apply constraints
        action = jnp.maximum(action, 0.0)
        action = jnp.minimum(action, X / self.config.dt)
        
        return action
    
    @property
    def name(self) -> str:
        return "Riccati Mean"
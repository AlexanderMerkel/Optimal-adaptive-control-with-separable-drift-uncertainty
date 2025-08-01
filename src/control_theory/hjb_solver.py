"""
Hamilton-Jacobi-Bellman (HJB) Neural PDE Solver for Optimal Control

Implements a neural network-based solver for the HJB equation:
∂V/∂t + max_u [drift terms + running cost] = 0

Where V(t,Y,X,p,α_l,α_h) is the value function.
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap
import flax.linen as nn
import optax
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .config import OptimalExecutionConfig, default_config
from .policies import Policy


@dataclass
class HJBConfig:
    """Configuration for HJB neural PDE solver."""
    
    # Network architecture
    hidden_dim: int = 128
    n_layers: int = 4
    activation: str = "tanh"
    
    # Training
    learning_rate: float = 1e-3
    n_epochs: int = 5000
    batch_size: int = 1024
    
    # Domain sampling
    n_interior: int = 8192   # Interior domain points
    n_boundary: int = 1024   # Boundary condition points
    
    # Loss weights
    pde_weight: float = 1.0
    boundary_weight: float = 10.0
    
    # Optimization bounds for control
    u_min: float = 0.0
    u_max: float = 100.0
    n_control_samples: int = 50


class ValueNetwork(nn.Module):
    """Neural network for value function V(t,Y,X,p,α_l,α_h)."""
    
    config: HJBConfig
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass through value network.
        
        Args:
            x: Input [t, Y, X, p, α_l, α_h] - shape (6,)
            
        Returns:
            V: Value function estimate - scalar
        """
        # Input normalization (helps training stability)
        x = nn.tanh(x / 10.0)  # Simple normalization
        
        # Deep feedforward network
        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.hidden_dim)(x)
            if self.config.activation == "tanh":
                x = nn.tanh(x)
            elif self.config.activation == "relu":
                x = nn.relu(x)
            else:
                raise ValueError(f"Unknown activation: {self.config.activation}")
        
        # Output layer (single value)
        V = nn.Dense(1)(x).squeeze()
        
        return V


class HJBOptimalPolicy(Policy):
    """Optimal policy derived from HJB value function."""
    
    def __init__(self, 
                 value_network: ValueNetwork,
                 value_params: Dict[str, Any],
                 problem_config: OptimalExecutionConfig = default_config,
                 hjb_config: HJBConfig = HJBConfig()):
        """Initialize HJB-based optimal policy."""
        self.value_network = value_network
        self.value_params = value_params
        self.problem_config = problem_config
        self.hjb_config = hjb_config
        
        # JIT compile value function and derivatives
        self._value_fn = jax.jit(self.value_network.apply)
        self._compute_optimal_control = jax.jit(self._compute_optimal_control_impl)
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute optimal control action using HJB solution."""
        return self._compute_optimal_control(state, time)
    
    def _compute_optimal_control_impl(self, state: jnp.ndarray, time: float) -> float:
        """Find optimal control by maximizing Hamiltonian."""
        # Create input for value network
        network_input = jnp.concatenate([jnp.array([time]), state])
        
        # Define objective function for control optimization
        def hamiltonian(u):
            return self._compute_hamiltonian(network_input, u)
        
        # Grid search for optimal control (simple but effective)
        u_grid = jnp.linspace(self.hjb_config.u_min, self.hjb_config.u_max, 
                             self.hjb_config.n_control_samples)
        
        # Evaluate Hamiltonian for all control values
        H_values = vmap(hamiltonian)(u_grid)
        
        # Find control that maximizes Hamiltonian
        optimal_idx = jnp.argmax(H_values)
        optimal_u = u_grid[optimal_idx]
        
        # Apply inventory constraint
        Y, X, p, alpha_l, alpha_h = state
        max_feasible_u = X / self.problem_config.dt
        optimal_u = jnp.minimum(optimal_u, max_feasible_u)
        optimal_u = jnp.maximum(optimal_u, 0.0)
        
        return optimal_u
    
    def _compute_hamiltonian(self, network_input: jnp.ndarray, u: float) -> float:
        """Compute Hamiltonian for given state and control."""
        t, Y, X, p, alpha_l, alpha_h = network_input
        
        # Value function and its derivatives
        V = self._value_fn({"params": self.value_params}, network_input)
        
        # Compute gradients
        grad_V = grad(lambda inp: self._value_fn({"params": self.value_params}, inp))(network_input)
        dV_dt, dV_dY, dV_dX, dV_dp, dV_dalpha_l, dV_dalpha_h = grad_V
        
        # Problem parameters
        lambda_l, kappa_l, lambda_h, kappa_h = self.problem_config.regime_params
        
        # Expected parameters based on belief
        expected_lambda = p * lambda_l + (1 - p) * lambda_h
        expected_kappa = p * kappa_l + (1 - p) * kappa_h
        expected_alpha = p * alpha_l + (1 - p) * alpha_h
        
        # Drift terms
        dY_drift = -expected_lambda * (u + expected_kappa * expected_alpha)
        dX_drift = -u
        
        # Alpha dynamics
        dalpha_l_drift = u + kappa_l * alpha_l
        dalpha_h_drift = u + kappa_h * alpha_h
        
        # Wonham filter drift (simplified - no stochastic term)
        f_low = lambda_l * (u + kappa_l * alpha_l)
        f_high = lambda_h * (u + kappa_h * alpha_h)
        # dp_drift would involve stochastic terms, set to 0 for deterministic part
        dp_drift = 0.0
        
        # Running cost
        running_cost = (Y - self.problem_config.rho * u) * u - self.problem_config.c * X**2
        
        # Hamiltonian (terms that don't involve control optimization)
        H = (dV_dY * dY_drift + 
             dV_dX * dX_drift + 
             dV_dp * dp_drift +
             dV_dalpha_l * dalpha_l_drift + 
             dV_dalpha_h * dalpha_h_drift + 
             running_cost)
        
        return H
    
    @property
    def name(self) -> str:
        return "HJB Optimal"


class HJBSolver:
    """Neural PDE solver for Hamilton-Jacobi-Bellman equation."""
    
    def __init__(self, 
                 problem_config: OptimalExecutionConfig = default_config,
                 hjb_config: HJBConfig = HJBConfig()):
        """Initialize HJB solver."""
        self.problem_config = problem_config
        self.hjb_config = hjb_config
        
        # Initialize network
        self.network = ValueNetwork(hjb_config)
        
        # Initialize optimizer
        self.optimizer = optax.adam(hjb_config.learning_rate)
        
        # JIT compile training functions
        self._pde_loss_fn = jax.jit(self._compute_pde_loss)
        self._boundary_loss_fn = jax.jit(self._compute_boundary_loss)
        self._update_step = jax.jit(self._update_step_impl)
    
    def solve(self, key: random.PRNGKey) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Solve HJB equation using neural PDE approach.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Tuple of (trained_params, training_history)
        """
        # Initialize network parameters
        dummy_input = jnp.ones(6)  # [t, Y, X, p, α_l, α_h]
        params = self.network.init(key, dummy_input)
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        # Training history
        history = {"pde_loss": [], "boundary_loss": [], "total_loss": []}
        
        print("Starting HJB neural PDE solver training...")
        
        for epoch in range(self.hjb_config.n_epochs):
            # Generate training data
            key, data_key = random.split(key)
            interior_points = self._sample_interior_points(data_key)
            boundary_points = self._sample_boundary_points(data_key)
            
            # Update step
            key, update_key = random.split(key)
            params, opt_state, losses = self._update_step(
                params, opt_state, interior_points, boundary_points, update_key
            )
            
            # Record history
            history["pde_loss"].append(float(losses["pde_loss"]))
            history["boundary_loss"].append(float(losses["boundary_loss"]))
            history["total_loss"].append(float(losses["total_loss"]))
            
            # Print progress
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1:4d}: PDE Loss = {losses['pde_loss']:.6f}, "
                      f"Boundary Loss = {losses['boundary_loss']:.6f}, "
                      f"Total Loss = {losses['total_loss']:.6f}")
        
        print("HJB solver training completed!")
        
        return params, history
    
    def _sample_interior_points(self, key: random.PRNGKey) -> jnp.ndarray:
        """Sample points in the interior domain."""
        keys = random.split(key, 6)
        
        # Sample each dimension
        t = random.uniform(keys[0], (self.hjb_config.n_interior,), minval=0.0, maxval=self.problem_config.T)
        Y = random.uniform(keys[1], (self.hjb_config.n_interior,), minval=90.0, maxval=110.0)
        X = random.uniform(keys[2], (self.hjb_config.n_interior,), minval=0.0, maxval=self.problem_config.X_0)
        p = random.uniform(keys[3], (self.hjb_config.n_interior,), minval=0.0, maxval=1.0)
        alpha_l = random.uniform(keys[4], (self.hjb_config.n_interior,), minval=0.0, maxval=5.0)
        alpha_h = random.uniform(keys[5], (self.hjb_config.n_interior,), minval=0.0, maxval=5.0)
        
        return jnp.stack([t, Y, X, p, alpha_l, alpha_h], axis=1)
    
    def _sample_boundary_points(self, key: random.PRNGKey) -> jnp.ndarray:
        """Sample points on the boundary (terminal condition)."""
        keys = random.split(key, 5)
        
        # Terminal time boundary (t = T)
        t = jnp.full((self.hjb_config.n_boundary,), self.problem_config.T)
        Y = random.uniform(keys[0], (self.hjb_config.n_boundary,), minval=90.0, maxval=110.0)  
        X = random.uniform(keys[1], (self.hjb_config.n_boundary,), minval=0.0, maxval=self.problem_config.X_0)
        p = random.uniform(keys[2], (self.hjb_config.n_boundary,), minval=0.0, maxval=1.0)
        alpha_l = random.uniform(keys[3], (self.hjb_config.n_boundary,), minval=0.0, maxval=5.0)
        alpha_h = random.uniform(keys[4], (self.hjb_config.n_boundary,), minval=0.0, maxval=5.0)
        
        return jnp.stack([t, Y, X, p, alpha_l, alpha_h], axis=1)
    
    def _compute_pde_loss(self, params: Dict[str, Any], points: jnp.ndarray) -> float:
        """Compute PDE residual loss."""
        def pde_residual(point):
            t, Y, X, p, alpha_l, alpha_h = point
            
            # Value function
            V = self.network.apply({"params": params}, point)
            
            # Time derivative
            dV_dt = grad(lambda inp: self.network.apply({"params": params}, inp))[0](point)
            
            # Find optimal control for this point
            def hamiltonian_at_point(u):
                # Simplified Hamiltonian for PDE residual
                running_cost = (Y - self.problem_config.rho * u) * u - self.problem_config.c * X**2
                return running_cost
            
            # Simple grid search for max Hamiltonian
            u_grid = jnp.linspace(0.0, min(50.0, X/self.problem_config.dt + 1e-6), 20)
            H_values = vmap(hamiltonian_at_point)(u_grid)
            max_H = jnp.max(H_values)
            
            # PDE residual: ∂V/∂t + max_u H = 0
            residual = dV_dt + max_H
            
            return residual**2
        
        return jnp.mean(vmap(pde_residual)(points))
    
    def _compute_boundary_loss(self, params: Dict[str, Any], boundary_points: jnp.ndarray) -> float:
        """Compute boundary condition loss."""
        def terminal_condition(point):
            t, Y, X, p, alpha_l, alpha_h = point
            
            # Value at terminal time
            V_pred = self.network.apply({"params": params}, point)
            
            # Terminal condition: V(T, Y, X, p, α_l, α_h) = Y*X - C*X²  
            V_true = Y * X - self.problem_config.C * X**2
            
            return (V_pred - V_true)**2
        
        return jnp.mean(vmap(terminal_condition)(boundary_points))
    
    def _update_step_impl(self, params, opt_state, interior_points, boundary_points, key):
        """Single training step."""
        def loss_fn(params):
            pde_loss = self._pde_loss_fn(params, interior_points)
            boundary_loss = self._boundary_loss_fn(params, boundary_points)
            total_loss = (self.hjb_config.pde_weight * pde_loss + 
                         self.hjb_config.boundary_weight * boundary_loss)
            return total_loss, {"pde_loss": pde_loss, "boundary_loss": boundary_loss, "total_loss": total_loss}
        
        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, losses
    
    def create_optimal_policy(self, params: Dict[str, Any]) -> HJBOptimalPolicy:
        """Create optimal policy from trained HJB solution."""
        return HJBOptimalPolicy(
            self.network, params, self.problem_config, self.hjb_config
        )


def train_hjb_optimal_control(problem_config: OptimalExecutionConfig = default_config,
                            hjb_config: HJBConfig = HJBConfig(),
                            key: random.PRNGKey = random.PRNGKey(42)) -> HJBOptimalPolicy:
    """
    Train HJB-based optimal control policy.
    
    Args:
        problem_config: Problem configuration
        hjb_config: HJB solver configuration  
        key: Random key for training
        
    Returns:
        Trained optimal policy
    """
    solver = HJBSolver(problem_config, hjb_config)
    params, history = solver.solve(key)
    optimal_policy = solver.create_optimal_policy(params)
    
    return optimal_policy
"""
Optimal Execution Environment with Regime Uncertainty

Implements the mathematical model from "Optimal adaptive control with separable drift uncertainty"
State: [Y, X, p, alpha_l, alpha_h] where:
- Y: Asset price  
- X: Inventory
- p: Belief state P(low regime | observations)
- alpha_l: Resilience state variable for low regime
- alpha_h: Resilience state variable for high regime
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional, NamedTuple

from .config import OptimalExecutionConfig, default_config


class StepResult(NamedTuple):
    """Result of one environment step."""
    next_state: jnp.ndarray    # Next state [Y, X, p, alpha_l, alpha_h]
    reward: float              # Instantaneous reward
    done: bool                 # Whether episode is finished
    info: dict                 # Additional information


class OptimalExecutionEnv:
    """
    Optimal execution environment with regime uncertainty.
    
    Implements the dynamics from the paper:
    - dY = -f(λ, κ, u, alpha) dt + σ dW  (price with regime-dependent impact)
    - dX = -u dt  (inventory)  
    - dp = Wonham filter update (belief)
    - dalpha_l = (u + κ_l alpha_l) dt  (low regime resilience)
    - dalpha_h = (u + κ_h alpha_h) dt  (high regime resilience)
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize environment with configuration."""
        self.config = config
        self.true_regime = None  # 0 = low, 1 = high (hidden state)
        
        # Pre-compute frequently used values
        self.sqrt_dt = jnp.sqrt(config.dt)
        
        # JIT compile step function for performance
        self._step_fn = jax.jit(self._step_impl)
        self._batch_step_fn = jax.vmap(self._step_impl, in_axes=(0, 0, 0, None))
    
    def reset(self, key: random.PRNGKey, batch_size: int = 1) -> jnp.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            key: Random key for regime initialization
            batch_size: Number of parallel environments
            
        Returns:
            Initial state(s) [Y, X, p, alpha_l, alpha_h]
        """
        if batch_size == 1:
            # Sample true regime (hidden from agent)
            self.true_regime = random.bernoulli(key, self.config.p_0).astype(jnp.float32)
            return self.config.initial_state
        else:
            # Batch case
            keys = random.split(key, batch_size)
            self.true_regime = random.bernoulli(keys, self.config.p_0).astype(jnp.float32)
            return jnp.tile(self.config.initial_state, (batch_size, 1))
    
    def step(self, state: jnp.ndarray, action: float, key: random.PRNGKey) -> StepResult:
        """
        Take one step in the environment.
        
        Args:
            state: Current state [Y, X, p, alpha_l, alpha_h]
            action: Control action (trading rate)
            key: Random key for noise generation
            
        Returns:
            StepResult with next state, reward, done flag, and info
        """
        return self._step_fn(state, action, key, self.true_regime)
    
    def batch_step(self, states: jnp.ndarray, actions: jnp.ndarray, key: random.PRNGKey) -> StepResult:
        """
        Take batch of steps in parallel environments.
        
        Args:
            states: Batch of current states [batch_size, 5]
            actions: Batch of actions [batch_size]
            key: Random key for noise generation
            
        Returns:
            StepResult with batched outputs
        """
        keys = random.split(key, states.shape[0])
        return self._batch_step_fn(states, actions, keys, self.true_regime)
    
    def _step_impl(self, state: jnp.ndarray, action: float, key: random.PRNGKey, 
                   true_regime: float) -> StepResult:
        """Internal step implementation (JIT compiled)."""
        Y, X, p, alpha_l, alpha_h = state
        u = action
        
        # Current regime parameters
        lambda_l, kappa_l, lambda_h, kappa_h = self.config.regime_params
        
        # True impact based on hidden regime
        true_lambda = (1 - true_regime) * lambda_l + true_regime * lambda_h
        true_kappa = (1 - true_regime) * kappa_l + true_regime * kappa_h
        true_alpha = (1 - true_regime) * alpha_l + true_regime * alpha_h
        
        # True drift for price (what actually happens)
        true_drift = -true_lambda * (u + true_kappa * true_alpha)
        
        # Generate price innovation
        dW = random.normal(key) * self.sqrt_dt
        dY = true_drift * self.config.dt + self.config.sigma * dW
        
        # Expected drift based on current belief (for filtering)
        # Each regime has its own alpha, so expected drift uses belief-weighted alphas
        expected_alpha = p * alpha_l + (1 - p) * alpha_h
        expected_lambda = p * lambda_l + (1 - p) * lambda_h
        expected_kappa = p * kappa_l + (1 - p) * kappa_h
        expected_drift = -expected_lambda * (u + expected_kappa * expected_alpha)
        
        # Innovation: observed change minus expected change
        innovation = dY - expected_drift * self.config.dt
        
        # Wonham filter update for belief
        # dp = (1/σ²) * p * (1-p) * [f_low - f_high] * innovation * dt
        f_low = lambda_l * (u + kappa_l * alpha_l)   # Low regime drift magnitude
        f_high = lambda_h * (u + kappa_h * alpha_h)  # High regime drift magnitude
        drift_difference = f_low - f_high
        
        dp = (1 / (self.config.sigma**2)) * p * (1 - p) * drift_difference * innovation * self.config.dt
        
        # Update state components
        Y_next = Y + dY
        X_next = X - u * self.config.dt  # Inventory decreases with trading
        p_next = jnp.clip(p + dp, 0.0, 1.0)  # Keep belief in [0,1]
        alpha_l_next = alpha_l + (u + kappa_l * alpha_l) * self.config.dt
        alpha_h_next = alpha_h + (u + kappa_h * alpha_h) * self.config.dt
        
        next_state = jnp.array([Y_next, X_next, p_next, alpha_l_next, alpha_h_next])
        
        # Compute instantaneous reward
        # J = E[∫(Y - ρu)u - cX² dt + YX - CX² (terminal)]
        execution_revenue = (Y - self.config.rho * u) * u * self.config.dt
        inventory_cost = -self.config.c * X**2 * self.config.dt
        reward = execution_revenue + inventory_cost
        
        # Check if done (inventory fully liquidated or time up)
        done = jnp.abs(X_next) < 1e-6  # Inventory essentially zero
        
        # Additional info for analysis
        info = {
            'true_regime': true_regime,
            'belief': p_next,
            'price': Y_next,
            'inventory': X_next,
            'execution_revenue': execution_revenue,
            'inventory_cost': inventory_cost,
            'resilience_l': alpha_l_next,
            'resilience_h': alpha_h_next,
            'innovation': innovation
        }
        
        return StepResult(next_state, reward, done, info)
    
    def compute_terminal_reward(self, state: jnp.ndarray) -> float:
        """
        Compute terminal reward for final state.
        
        Args:
            state: Final state [Y, X, p, alpha_l, alpha_h]
            
        Returns:
            Terminal reward: Y*X - C*X²
        """
        Y, X, _, _, _ = state
        liquidation_value = Y * X
        terminal_penalty = -self.config.C * X**2
        return liquidation_value + terminal_penalty
    
    def generate_trajectory(self, policy_fn, n_steps: Optional[int] = None, 
                          key: random.PRNGKey = random.PRNGKey(0)) -> dict:
        """
        Generate complete trajectory using given policy.
        
        Args:
            policy_fn: Function that maps (state, time) -> action
            n_steps: Number of steps (uses config default if None)
            key: Random key for trajectory generation
            
        Returns:
            Dictionary with trajectory data
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        
        # Initialize
        key, reset_key = random.split(key)
        state = self.reset(reset_key)
        
        # Storage
        states = jnp.zeros((n_steps + 1, 5))
        actions = jnp.zeros(n_steps) 
        rewards = jnp.zeros(n_steps)
        infos = []
        
        states = states.at[0].set(state)
        
        # Generate trajectory
        for t in range(n_steps):
            current_time = t * self.config.dt
            
            # Get action from policy
            action = policy_fn(state, current_time)
            actions = actions.at[t].set(action)
            
            # Take step
            key, step_key = random.split(key)
            result = self.step(state, action, step_key)
            
            # Store results
            state = result.next_state
            states = states.at[t + 1].set(state)
            rewards = rewards.at[t].set(result.reward)
            infos.append(result.info)
            
            if result.done:
                break
        
        # Add terminal reward
        terminal_reward = self.compute_terminal_reward(state)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'terminal_reward': terminal_reward,
            'total_reward': jnp.sum(rewards) + terminal_reward,
            'infos': infos,
            'final_state': state,
            'n_steps': n_steps
        }
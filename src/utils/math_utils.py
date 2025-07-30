"""
Mathematical Utilities for Optimal Execution System

This module provides common mathematical operations, transformations, and utilities
used across the optimal execution controllers. It consolidates shared mathematical
logic including Wonham filtering, state transformations, and statistical operations.

Key Components:
- Wonham filter implementation for regime detection
- State space transformations and bounds checking
- Price dynamics and innovation processes
- Statistical analysis utilities
- Probability and belief state operations

Usage:
    from utils.math_utils import WonhamFilter, StateManager, PriceDynamics
    from utils.config import get_config

    config = get_config()
    filter = WonhamFilter(config)
    state_mgr = StateManager(config)

    new_belief = filter.update_belief(current_belief, innovation, action, A_l, A_h)
"""

from typing import Tuple, Optional
import warnings

import jax
import jax.numpy as jnp
from jax import random

from .config import Config


class WonhamFilter:
    """Wonham filter for regime detection under uncertainty."""

    def __init__(self, config: Config):
        """Initialize Wonham filter with configuration parameters.

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.SIGMA = config.SIGMA
        self.LAMBDA_L = config.LAMBDA_L
        self.LAMBDA_H = config.LAMBDA_H
        self.KAPPA_L = config.KAPPA_L
        self.KAPPA_H = config.KAPPA_H
        self.dt = config.dt
        self.inv_sigma = 1.0 / self.SIGMA  # Precompute for efficiency

    def update_belief(
        self,
        belief: jnp.ndarray,
        innovation: jnp.ndarray,
        actions: jnp.ndarray,
        A_l: jnp.ndarray,
        A_h: jnp.ndarray
    ) -> jnp.ndarray:
        """Update belief state using Wonham filter.

        Args:
            belief: Current belief p(t) that true regime is low
            innovation: Innovation process (prediction error + noise)
            actions: Control actions taken
            A_l: Accumulator state for low regime
            A_h: Accumulator state for high regime

        Returns:
            Updated belief state
        """
        regime_diff = (self.LAMBDA_L * (actions + self.KAPPA_L * A_l) -
                      self.LAMBDA_H * (actions + self.KAPPA_H * A_h))

        # Wonham filter update equation
        dp = -self.inv_sigma * belief * (1.0 - belief) * regime_diff * innovation * self.dt

        new_belief = belief + dp
        return jnp.clip(new_belief, 1e-6, 1.0 - 1e-6)

    def compute_innovation(
        self,
        true_regime: jnp.ndarray,
        expected_drift: jnp.ndarray,
        dW: jnp.ndarray,
        actions: jnp.ndarray,
        A_l: jnp.ndarray,
        A_h: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute innovation process and prediction error.

        Args:
            true_regime: True regime indicator (0=low, 1=high)
            expected_drift: Expected price drift based on belief
            dW: Brownian motion increment
            actions: Control actions
            A_l: Low regime accumulator
            A_h: High regime accumulator

        Returns:
            Tuple of (innovation, prediction_error)
        """
        actual_drift = jnp.where(
            true_regime == 0,
            self.LAMBDA_L * (actions + self.KAPPA_L * A_l),
            self.LAMBDA_H * (actions + self.KAPPA_H * A_h)
        )

        prediction_error = (actual_drift - expected_drift) * self.dt
        innovation = prediction_error / self.SIGMA + dW / jnp.sqrt(self.dt)
        return innovation, prediction_error

    def compute_expected_drift(
        self,
        belief: jnp.ndarray,
        actions: jnp.ndarray,
        A_l: jnp.ndarray,
        A_h: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute expected price drift based on current belief.

        Args:
            belief: Current belief state
            actions: Control actions
            A_l: Low regime accumulator
            A_h: High regime accumulator

        Returns:
            Expected drift
        """
        drift_low = self.LAMBDA_L * (actions + self.KAPPA_L * A_l)
        drift_high = self.LAMBDA_H * (actions + self.KAPPA_H * A_h)

        return belief * drift_low + (1.0 - belief) * drift_high


class StateManager:
    """State space management and transformations."""

    def __init__(self, config: Config):
        """Initialize state manager with configuration.

        Args:
            config: Configuration object containing state parameters
        """
        self.config = config
        self.low_bounds = config.low_bounds
        self.high_bounds = config.high_bounds
        self.initial_state = config.initial_state_array
        self.dt = config.dt

        # State dimension information
        self.STATE_DIM = config.STATE_DIM
        self.INTERNAL_DIM = config.INTERNAL_DIM
        self.STATE_NAMES = config.STATE_NAMES

    def clip_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Clip state to valid bounds.

        Args:
            state: State array to clip

        Returns:
            Clipped state array
        """
        return jnp.clip(state, self.low_bounds, self.high_bounds)

    def initialize_batch(self, batch_size: int, key: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize batch of states with random true regimes.

        Args:
            batch_size: Number of trajectories to initialize
            key: Random key for regime sampling

        Returns:
            Tuple of (internal_states, observable_states)
        """
        if key is None:
            key = random.PRNGKey(42)

        # Sample true regimes (0=low, 1=high)
        regime_key, _ = random.split(key)
        true_regimes = random.bernoulli(regime_key, 0.5, (batch_size,)).astype(jnp.float32)

        observable_states = jnp.tile(self.initial_state, (batch_size, 1))

        internal_states = jnp.column_stack([observable_states, true_regimes])

        return internal_states, observable_states

    def extract_state_components(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """Extract individual state components.

        Args:
            state: State array of shape (..., 6) for observable or (..., 7) for internal

        Returns:
            Tuple of state components (t, S, X, p, A_l, A_h, [regime])
        """
        if state.shape[-1] == self.STATE_DIM:  # Observable state
            return tuple(state[..., i] for i in range(self.STATE_DIM))
        elif state.shape[-1] == self.INTERNAL_DIM:  # Internal state
            return tuple(state[..., i] for i in range(self.INTERNAL_DIM))
        else:
            raise ValueError(f"Invalid state dimension: {state.shape[-1]}")

    def update_accumulators(
        self,
        A_l: jnp.ndarray,
        A_h: jnp.ndarray,
        actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update accumulator states.

        Args:
            A_l: Low regime accumulator
            A_h: High regime accumulator
            actions: Control actions

        Returns:
            Updated (A_l_next, A_h_next)
        """
        A_l_next = A_l + (actions + self.config.KAPPA_L * A_l) * self.dt
        A_h_next = A_h + (actions + self.config.KAPPA_H * A_h) * self.dt

        return A_l_next, A_h_next

    def validate_state_bounds(self, state: jnp.ndarray, warn: bool = True) -> bool:
        """Validate that state is within bounds.

        Args:
            state: State to validate
            warn: Whether to issue warnings for out-of-bounds values

        Returns:
            True if state is within bounds
        """
        if state.shape[-1] > len(self.low_bounds):
            # For internal state, only check observable components
            state = state[..., :len(self.low_bounds)]

        within_bounds = jnp.all((state >= self.low_bounds) & (state <= self.high_bounds))

        if not within_bounds and warn:
            violations = jnp.sum((state < self.low_bounds) | (state > self.high_bounds))
            warnings.warn(f"State bounds violated in {violations} components")

        return bool(within_bounds)


class PriceDynamics:
    """Price dynamics and market evolution utilities."""

    def __init__(self, config: Config):
        """Initialize price dynamics with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.SIGMA = config.SIGMA
        self.RHO = config.RHO
        self.C_RUNNING = config.C_RUNNING
        self.dt = config.dt

        # Precompute square root of dt for efficiency
        self.sqrt_dt = jnp.sqrt(self.dt)

    def compute_price_increment(
        self,
        actual_drift: jnp.ndarray,
        dW: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute price increment dS.

        Args:
            actual_drift: Actual drift based on true regime
            dW: Brownian motion increment

        Returns:
            Price increment dS
        """
        return actual_drift * self.dt + self.SIGMA * dW

    def compute_reward(
        self,
        S: jnp.ndarray,
        X: jnp.ndarray,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute instantaneous reward.

        Args:
            S: Current price
            X: Current inventory
            actions: Control actions

        Returns:
            Instantaneous reward
        """
        return ((S - self.RHO * actions) * actions - self.C_RUNNING * X**2) * self.dt

    def compute_terminal_reward(
        self,
        S_final: jnp.ndarray,
        X_final: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute terminal reward.

        Args:
            S_final: Final price
            X_final: Final inventory

        Returns:
            Terminal reward
        """
        return S_final * X_final - self.config.C_TERMINAL * X_final**2


class StatisticalUtils:
    """Statistical analysis utilities."""

    @staticmethod
    def compute_regime_accuracy(
        final_beliefs: jnp.ndarray,
        true_regimes: jnp.ndarray
    ) -> float:
        """Compute regime detection accuracy.

        Args:
            final_beliefs: Final belief states
            true_regimes: True regime indicators

        Returns:
            Accuracy as fraction of correct classifications
        """
        predicted_regimes = (final_beliefs < 0.5).astype(int)
        true_regimes_int = true_regimes.astype(int)
        return float(jnp.mean(predicted_regimes == true_regimes_int))

    @staticmethod
    def compute_performance_metrics(rewards: jnp.ndarray) -> dict:
        """Compute performance metrics for reward array.

        Args:
            rewards: Array of total rewards/profits

        Returns:
            Dictionary of performance metrics
        """
        return {
            'mean': float(jnp.mean(rewards)),
            'std': float(jnp.std(rewards)),
            'min': float(jnp.min(rewards)),
            'max': float(jnp.max(rewards)),
            'median': float(jnp.median(rewards)),
            'q25': float(jnp.percentile(rewards, 25)),
            'q75': float(jnp.percentile(rewards, 75))
        }

    @staticmethod
    def sharpe_ratio(returns: jnp.ndarray, risk_free_rate: float = 0.0) -> float:
        """Compute Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (default 0)

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate
        return float(jnp.mean(excess_returns) / jnp.std(excess_returns))

    @staticmethod
    def max_drawdown(cumulative_returns: jnp.ndarray) -> float:
        """Compute maximum drawdown.

        Args:
            cumulative_returns: Cumulative returns over time

        Returns:
            Maximum drawdown as positive value
        """
        running_max = jnp.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        return float(jnp.max(drawdown))


# Factory functions for convenience
def create_wonham_filter(config: Optional[Config] = None) -> WonhamFilter:
    """Create Wonham filter with configuration."""
    if config is None:
        from .config import get_config
        config = get_config()
    return WonhamFilter(config)


def create_state_manager(config: Optional[Config] = None) -> StateManager:
    """Create state manager with configuration."""
    if config is None:
        from .config import get_config
        config = get_config()
    return StateManager(config)


def create_price_dynamics(config: Optional[Config] = None) -> PriceDynamics:
    """Create price dynamics with configuration."""
    if config is None:
        from .config import get_config
        config = get_config()
    return PriceDynamics(config)


# Convenience functions
def update_full_state(
    internal_state: jnp.ndarray,
    actions: jnp.ndarray,
    dW: jnp.ndarray,
    config: Optional[Config] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Update full internal state with all dynamics.

    This is a consolidated state update function that can be used by controllers
    to avoid duplicating the complex state evolution logic.

    Args:
        internal_state: Current internal state [t, S, X, p, A_l, A_h, regime]
        actions: Control actions
        dW: Brownian motion increments
        config: Configuration object

    Returns:
        Tuple of (next_internal, next_observable, rewards, info)
    """
    if config is None:
        from .config import get_config
        config = get_config()

    wonham_filter = WonhamFilter(config)
    state_manager = StateManager(config)
    price_dynamics = PriceDynamics(config)

    t, S, X, p, A_l, A_h, true_regime = state_manager.extract_state_components(internal_state)

    t_next = t + config.dt

    X_next = X - actions * config.dt

    A_l_next, A_h_next = state_manager.update_accumulators(A_l, A_h, actions)

    expected_drift = wonham_filter.compute_expected_drift(p, actions, A_l, A_h)

    innovation, prediction_error = wonham_filter.compute_innovation(
        true_regime, expected_drift, dW, actions, A_l, A_h
    )

    actual_drift = jnp.where(
        true_regime == 0,
        config.LAMBDA_L * (actions + config.KAPPA_L * A_l),
        config.LAMBDA_H * (actions + config.KAPPA_H * A_h)
    )
    dS = price_dynamics.compute_price_increment(actual_drift, dW)
    S_next = S + dS

    p_next = wonham_filter.update_belief(p, innovation, actions, A_l, A_h)

    # Construct next states
    next_internal = jnp.stack([t_next, S_next, X_next, p_next, A_l_next, A_h_next, true_regime], axis=1)
    next_observable = next_internal[:, :6]

    # Clip observable state to bounds
    next_observable = state_manager.clip_state(next_observable)
    next_internal = next_internal.at[:, :6].set(next_observable)

    rewards = price_dynamics.compute_reward(S_next, X_next, actions)

    info = {
        'innovation': innovation,
        'prediction_error': prediction_error,
        'true_regime': true_regime,
        'actual_drift': actual_drift,
        'expected_drift': expected_drift
    }

    return next_internal, next_observable, rewards, info


if __name__ == "__main__":
    # Test mathematical utilities
    from .config import get_config
    import jax.random as random

    config = get_config()

    # Test Wonham filter
    print("Testing Wonham Filter...")
    wonham_filter = WonhamFilter(config)

    # Test state manager
    print("Testing State Manager...")
    state_manager = StateManager(config)
    key = random.PRNGKey(42)
    internal_states, obs_states = state_manager.initialize_batch(5, key)
    print(f"Initialized batch shapes: internal={internal_states.shape}, observable={obs_states.shape}")

    # Test price dynamics
    print("Testing Price Dynamics...")
    price_dynamics = PriceDynamics(config)

    # Test full state update
    print("Testing Full State Update...")
    actions = jnp.ones((5,)) * 0.5
    dW = random.normal(key, (5,))

    next_internal, next_obs, rewards, info = update_full_state(internal_states, actions, dW, config)
    print(f"State update successful: rewards mean = {jnp.mean(rewards):.4f}")

    print("All mathematical utilities tests passed!")

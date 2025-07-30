"""
Base Controller Class for Optimal Execution Methods

This provides a common foundation for all optimal execution controllers including:
- Shared parameter initialization using centralized configuration
- Common environment dynamics (Wonham filtering, state evolution)
- Standardized evaluation interface using shared utilities
- Mathematical utilities for state management and filtering

All concrete controllers should inherit from this base class.
"""

# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Local imports
from ..utils import get_config, WonhamFilter, StateManager, PriceDynamics, StatisticalUtils


class BaseOptimalExecutionController(ABC):
    """Abstract base class for optimal execution controllers."""

    def __init__(self, config=None):
        """Initialize with centralized configuration."""
        self.config = config if config is not None else get_config()

        self.wonham_filter = WonhamFilter(self.config)
        self.state_manager = StateManager(self.config)
        self.price_dynamics = PriceDynamics(self.config)

        # Expose frequently used parameters for backward compatibility
        self.T = self.config.T
        self.N = self.config.N
        self.dt = self.config.dt
        self.SQRT_DT = jnp.sqrt(self.dt)

        # Market parameters
        self.SIGMA = self.config.SIGMA
        self.RHO = self.config.RHO
        self.C_RUNNING = self.config.C_RUNNING
        self.C_TERMINAL = self.config.C_TERMINAL

        # Regime parameters
        self.LAMBDA_L = self.config.LAMBDA_L
        self.LAMBDA_H = self.config.LAMBDA_H
        self.KAPPA_L = self.config.KAPPA_L
        self.KAPPA_H = self.config.KAPPA_H

        # State management
        self.low_bounds = self.config.low_bounds
        self.high_bounds = self.config.high_bounds

    def reset_env_with_true_regime(self, key, batch_size=64):
        """Initialize environment with hidden true regime per trajectory."""
        internal_state, observable_state = self.state_manager.initialize_batch(batch_size, key)
        return internal_state, observable_state

    def step_env_with_innovations(self, key, internal_state, actions, batch_size=64):
        """Environment step implementing rigorous Wonham filtering with innovations process."""
        key, subkey = random.split(key)
        dW = self.SQRT_DT * random.normal(subkey, (batch_size,), dtype=jnp.float32)
        actions = actions.ravel()[:batch_size]

        t, S, X, p, A_l, A_h, true_regime = self.state_manager.extract_state_components(internal_state)

        expected_drift = self.wonham_filter.compute_expected_drift(p, actions, A_l, A_h)

        innovation, prediction_error = self.wonham_filter.compute_innovation(
            true_regime, expected_drift, dW, actions, A_l, A_h
        )

        t_next = t + self.dt
        X_next = X - actions * self.dt

        A_l_next, A_h_next = self.state_manager.update_accumulators(A_l, A_h, actions)

        actual_drift = jnp.where(
            true_regime == 0,
            -self.LAMBDA_L * (actions + self.KAPPA_L * A_l),
            -self.LAMBDA_H * (actions + self.KAPPA_H * A_h)
        )
        dS = self.price_dynamics.compute_price_increment(actual_drift, dW)
        S_next = S + dS

        p_next = self.wonham_filter.update_belief(p, innovation, actions, A_l, A_h)

        # Construct next states
        next_internal = jnp.stack([t_next, S_next, X_next, p_next, A_l_next, A_h_next, true_regime], axis=1)
        next_observable = next_internal[:, :6]

        # Clip observable state to bounds
        next_observable = self.state_manager.clip_state(next_observable)
        next_internal = next_internal.at[:, :6].set(next_observable)

        reward = self.price_dynamics.compute_reward(S_next, X_next, actions)

        info = {
            "innovation": innovation,
            "true_regime": true_regime,
            "prediction_error": prediction_error,
        }

        return next_internal, next_observable, reward, jnp.zeros(batch_size, dtype=bool), info

    @abstractmethod
    def compute_control_action(self, observable_state, time_step=None):
        """Compute control action based on observable state.

        Args:
            observable_state: 6D observable state [t, S, X, p, A_l, A_h]
            time_step: Optional time step index

        Returns:
            Control actions for the batch
        """
        pass

    def evaluate_performance(self, key, num_trajectories=100, n_steps=200, **kwargs):
        """
        Standard evaluation interface for all controllers.

        Args:
            key: JAX random key
            num_trajectories: Number of trajectories to evaluate
            n_steps: Number of time steps per trajectory
            **kwargs: Method-specific arguments

        Returns:
            Dictionary with standardized performance metrics
        """
        key, reset_key, rollout_key = random.split(key, 3)

        initial_internal, _ = self.reset_env_with_true_regime(reset_key, num_trajectories)

        # Run controller
        states, actions, rewards, final_internal, infos = self._rollout(
            rollout_key, initial_internal, n_steps, num_trajectories, **kwargs
        )

        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        true_regimes_np = np.array(final_internal[:, 6])

        # Performance metrics using StatisticalUtils
        total_profits = np.sum(rewards_np, axis=0)
        beliefs = states_np[:, :, 3]  # p(t)
        final_beliefs = beliefs[-1, :]

        # Regime detection accuracy using StatisticalUtils
        accuracy = StatisticalUtils.compute_regime_accuracy(
            jnp.array(final_beliefs), jnp.array(true_regimes_np)
        )

        # Performance metrics using StatisticalUtils
        profit_metrics = StatisticalUtils.compute_performance_metrics(jnp.array(total_profits))

        return {
            "method": self.__class__.__name__,
            "total_profits": total_profits,
            "mean_profit": profit_metrics['mean'],
            "std_profit": profit_metrics['std'],
            "regime_accuracy": accuracy,
            "states": states_np,
            "actions": actions_np,
            "rewards": rewards_np,
            "true_regimes": true_regimes_np,
            "beliefs": beliefs,
            "profit_metrics": profit_metrics,  # Additional detailed metrics
        }

    def _rollout(self, key, initial_internal, n_steps=200, batch_size=64, **kwargs):
        """Internal rollout method using the controller's compute_control_action method."""

        def scan_body(carry, time_step):
            key, internal_state = carry
            key, env_key = random.split(key)

            # Agent observes only 6D state (no true regime)
            observable = internal_state[:, :6]

            # Controller-specific control decision
            actions = self.compute_control_action(observable, time_step)

            # Environment step with filtering
            next_internal, next_obs, reward, done, info = self.step_env_with_innovations(
                env_key, internal_state, actions, batch_size
            )

            return (key, next_internal), (observable, actions, reward, info)

        time_steps = jnp.arange(n_steps)
        (_, final_internal), (states, actions, rewards, infos) = jax.lax.scan(
            scan_body, (key, initial_internal), time_steps
        )

        # Terminal reward using price dynamics
        final_obs = final_internal[:, :6]
        S_final, X_final = final_obs[:, 1], final_obs[:, 2]
        terminal_reward = self.price_dynamics.compute_terminal_reward(S_final, X_final)
        rewards = rewards.at[-1, :].add(terminal_reward)

        return states, actions, rewards, final_internal, infos

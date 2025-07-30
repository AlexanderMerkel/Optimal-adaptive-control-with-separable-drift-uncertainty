"""
Centralized Configuration Manager for Optimal Execution System

This module provides a singleton configuration manager that handles all parameter
loading, validation, and access across the entire codebase. This eliminates the
duplicate YAML loading and parameter extraction found in multiple controllers.

Usage:
    from utils.config import Config

    config = Config()
    T = config.T
    lambda_l = config.LAMBDA_L

    # Or get all parameters as dict
    params = config.get_all()
"""

# Standard library imports
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import threading

# Third-party imports
import jax.numpy as jnp
import numpy as np


class Config:
    """Singleton configuration manager for centralized parameter access."""

    _instance: Optional['Config'] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls) -> 'Config':
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, params_path: Optional[str] = None):
        """Initialize configuration manager (only once due to singleton)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Determine parameter file path
            if params_path is None:
                # Default path relative to this file
                params_path = Path(__file__).parent.parent / "model_parameters.yaml"
            else:
                params_path = Path(params_path)

            if not params_path.exists():
                raise FileNotFoundError(f"Model parameters file not found: {params_path}")

            # Load and parse parameters
            with open(params_path, "r") as file:
                self._raw_params = yaml.safe_load(file)

            # Initialize core parameters
            self._initialize_parameters()
            self._initialize_derived_parameters()
            self._initialize_state_management()

            self._initialized = True

    def _initialize_parameters(self):
        """Initialize core model parameters from YAML."""
        # Time parameters
        self.T = float(self._raw_params["T"])
        self.N = 200  # Standard time steps across all controllers
        self.dt = self.T / self.N
        self.SQRT_DT = jnp.sqrt(self.dt)

        # Market parameters
        self.SIGMA = float(self._raw_params["SIGMA"])
        self.RHO = float(self._raw_params["RHO"])

        # Cost parameters
        self.C_RUNNING = float(self._raw_params["C_RUNNING"])
        self.C_TERMINAL = float(self._raw_params["C_TERMINAL"])

        # Regime parameters
        self.LAMBDA_L = float(self._raw_params["LAMBDA_L"])
        self.LAMBDA_H = float(self._raw_params["LAMBDA_H"])
        self.KAPPA_L = float(self._raw_params["KAPPA_L"])
        self.KAPPA_H = float(self._raw_params["KAPPA_H"])

        # State bounds and initial conditions
        self.STATE_BOUNDS = self._raw_params["STATE_BOUNDS"]
        self.INITIAL_STATE = self._raw_params["INITIAL_STATE"]

    def _initialize_derived_parameters(self):
        """Initialize commonly used derived parameters."""
        # Mean regime parameters (used by multiple controllers)
        self.LAMBDA_MEAN = 0.5 * (self.LAMBDA_L + self.LAMBDA_H)
        self.KAPPA_MEAN = 0.5 * (self.KAPPA_L + self.KAPPA_H)

        # Time grid
        self.time_grid = np.linspace(0.0, self.T, self.N + 1)

        # State space bounds as arrays
        self.low_bounds = jnp.array([
            v[0] for v in self.STATE_BOUNDS.values()
        ], dtype=jnp.float32)

        self.high_bounds = jnp.array([
            v[1] for v in self.STATE_BOUNDS.values()
        ], dtype=jnp.float32)

        # Initial state as array
        self.initial_state_array = jnp.array([
            self.INITIAL_STATE[k] for k in ["t", "S", "X", "p", "A_l", "A_h"]
        ], dtype=jnp.float32)

    def _initialize_state_management(self):
        """Initialize state space configuration."""
        # State dimension information
        self.STATE_DIM = 6  # Observable state dimension
        self.INTERNAL_DIM = 7  # Internal state includes true regime

        # State variable names for reference
        self.STATE_NAMES = ["t", "S", "X", "p", "A_l", "A_h"]
        self.INTERNAL_STATE_NAMES = ["t", "S", "X", "p", "A_l", "A_h", "regime"]

    def get_all(self) -> Dict[str, Any]:
        """Get all parameters as dictionary for compatibility."""
        return {
            # Core parameters
            "T": self.T,
            "N": self.N,
            "dt": self.dt,
            "SIGMA": self.SIGMA,
            "RHO": self.RHO,
            "C_RUNNING": self.C_RUNNING,
            "C_TERMINAL": self.C_TERMINAL,
            "LAMBDA_L": self.LAMBDA_L,
            "LAMBDA_H": self.LAMBDA_H,
            "KAPPA_L": self.KAPPA_L,
            "KAPPA_H": self.KAPPA_H,
            "STATE_BOUNDS": self.STATE_BOUNDS,
            "INITIAL_STATE": self.INITIAL_STATE,

            # Derived parameters
            "LAMBDA_MEAN": self.LAMBDA_MEAN,
            "KAPPA_MEAN": self.KAPPA_MEAN,
            "low_bounds": self.low_bounds,
            "high_bounds": self.high_bounds,
            "initial_state_array": self.initial_state_array,
            "time_grid": self.time_grid,
        }

    def get_regime_parameters(self, regime_type: str = "mean") -> Dict[str, float]:
        """Get regime-specific parameters."""
        if regime_type == "mean":
            return {
                "lambda": self.LAMBDA_MEAN,
                "kappa": self.KAPPA_MEAN
            }
        elif regime_type == "low":
            return {
                "lambda": self.LAMBDA_L,
                "kappa": self.KAPPA_L
            }
        elif regime_type == "high":
            return {
                "lambda": self.LAMBDA_H,
                "kappa": self.KAPPA_H
            }
        else:
            raise ValueError(f"Invalid regime_type: {regime_type}. Must be 'mean', 'low', or 'high'")

    def validate_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Validate and clip state to bounds."""
        return jnp.clip(state, self.low_bounds, self.high_bounds)

    def print_summary(self):
        """Print configuration summary for debugging."""
        print("=== Optimal Execution Configuration ===")
        print(f"Time horizon: T = {self.T}")
        print(f"Time steps: N = {self.N} (dt = {self.dt:.6f})")
        print(f"Market volatility: σ = {self.SIGMA}")
        print(f"Temporary impact: ρ = {self.RHO}")
        print(f"Cost parameters: c = {self.C_RUNNING}, C = {self.C_TERMINAL}")
        print(f"Regime parameters:")
        print(f"  Low regime: λ_L = {self.LAMBDA_L}, κ_L = {self.KAPPA_L}")
        print(f"  High regime: λ_H = {self.LAMBDA_H}, κ_H = {self.KAPPA_H}")
        print(f"  Mean: λ_mean = {self.LAMBDA_MEAN:.4f}, κ_mean = {self.KAPPA_MEAN:.4f}")
        print("=" * 40)


# Global instance for easy access
_config_instance: Optional[Config] = None

def get_config(params_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        # If a custom path is provided and config isn't initialized yet, we need to reinitialize
        if params_path is not None and not _config_instance._initialized:
            _config_instance.__init__(params_path)
    return _config_instance


def reset_config():
    """Reset configuration for testing purposes."""
    global _config_instance
    Config._instance = None
    Config._initialized = False
    _config_instance = None


# Convenience functions for common access patterns
def get_time_parameters():
    """Get time-related parameters."""
    config = get_config()
    return config.T, config.N, config.dt, config.time_grid

def get_market_parameters():
    """Get market-related parameters."""
    config = get_config()
    return config.SIGMA, config.RHO

def get_cost_parameters():
    """Get cost-related parameters."""
    config = get_config()
    return config.C_RUNNING, config.C_TERMINAL

def get_regime_parameters(regime_type: str = "mean"):
    """Get regime-specific parameters."""
    config = get_config()
    return config.get_regime_parameters(regime_type)

def get_state_bounds():
    """Get state bounds."""
    config = get_config()
    return config.low_bounds, config.high_bounds

def get_initial_state():
    """Get initial state array."""
    config = get_config()
    return config.initial_state_array


if __name__ == "__main__":
    # Test the configuration manager
    config = Config()
    config.print_summary()

    # Test convenience functions
    T, N, dt, time_grid = get_time_parameters()
    print(f"Time parameters: T={T}, N={N}, dt={dt:.6f}")

    regime_params = get_regime_parameters("low")
    print(f"Low regime: λ={regime_params['lambda']}, κ={regime_params['kappa']}")

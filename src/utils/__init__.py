"""
Shared Utilities Package for Optimal Execution System

This package contains common utilities, configurations, and mathematical functions
that are shared across all controllers and components in the optimal execution system.

Modules:
    config: Centralized configuration management
    riccati_solver: Common Riccati equation solving utilities
    math_utils: Mathematical operations and transformations
    networks: Shared neural network architectures
    evaluation: Unified evaluation framework
"""

from .config import Config, get_config, get_time_parameters, get_market_parameters, get_cost_parameters, get_regime_parameters, get_state_bounds, get_initial_state
from .riccati_solver import RiccatiSolver, create_riccati_solver, solve_single_riccati, solve_riccati_grid
from .math_utils import WonhamFilter, StateManager, PriceDynamics, StatisticalUtils, create_wonham_filter, create_state_manager, create_price_dynamics, update_full_state

__all__ = [
    "Config",
    "get_config",
    "get_time_parameters",
    "get_market_parameters",
    "get_cost_parameters",
    "get_regime_parameters",
    "get_state_bounds",
    "get_initial_state",
    "RiccatiSolver",
    "create_riccati_solver",
    "solve_single_riccati",
    "solve_riccati_grid",
    "WonhamFilter",
    "StateManager",
    "PriceDynamics",
    "StatisticalUtils",
    "create_wonham_filter",
    "create_state_manager",
    "create_price_dynamics",
    "update_full_state",
]

"""
Control Theory Framework

A general-purpose framework for stochastic optimal control problems, built on the 
foundation of the optimal execution system. This package provides abstractions for:

- Control policies (deterministic, stochastic, neural)
- State transition systems and environments  
- Value function solvers (Riccati, HJB, neural)
- Trajectory generation and analysis
- Performance evaluation and metrics

The framework is designed for extensibility while maintaining mathematical rigor
and computational efficiency through JAX integration.

Core Components:
    - State, Action: Immutable data structures for control problems
    - ControlPolicy: Abstract interface for control computation
    - StateTransitionSystem: Abstract interface for dynamics
    - ControlEnvironment: Complete environment abstraction
    - ValueFunctionSolver: Abstract interface for value function computation

Usage:
    from control_theory import ControlPolicy, RiccatiPolicy
    from control_theory import ControlEnvironment, OptimalExecutionEnvironment
    from control_theory import TrajectoryGenerator, ControlSystemFactory
"""

from .core import (
    State,
    Action, 
    Reward,
    Info,
    ControlPolicy,
    StateTransitionSystem,
    RewardFunction
)

from .policies import (
    RiccatiPolicy,
    NeuralPolicy,
    NetworkArchitectureRegistry
)

from .value_functions import (
    ValueFunctionSolver,
    ValueFunction,
    GeneralRiccatiSolver
)

from .environments import (
    ControlEnvironment,
    OptimalExecutionEnvironment,
    StateSpace,
    NoiseModel,
    BrownianMotion
)

from .trajectory import (
    Trajectory,
    BatchTrajectories,
    TrajectoryGenerator,
    TrajectoryAnalyzer,
    generate_trajectory_from_policy,
    generate_batch_trajectories_from_policy
)

from .factory import (
    ControlSystemFactory,
    ControlSystemConfig,
    ControlSystem,
    get_factory,
    create_neumann_voss_system,
    create_ce_system,
    create_neural_system
)

__version__ = "0.1.0"

__all__ = [
    # Core types and interfaces
    "State",
    "Action", 
    "Reward",
    "Info",
    "ControlPolicy",
    "StateTransitionSystem", 
    "RewardFunction",
    
    # Policy implementations
    "RiccatiPolicy",
    "NeuralPolicy",
    "NetworkArchitectureRegistry",
    
    # Value function solvers
    "ValueFunctionSolver",
    "ValueFunction",
    "GeneralRiccatiSolver",
    
    # Environment and dynamics
    "ControlEnvironment",
    "OptimalExecutionEnvironment", 
    "StateSpace",
    "NoiseModel",
    "BrownianMotion",
    
    # Trajectory generation and analysis
    "Trajectory",
    "BatchTrajectories",
    "TrajectoryGenerator",
    "TrajectoryAnalyzer",
    "generate_trajectory_from_policy",
    "generate_batch_trajectories_from_policy",
    
    # Control system factory and utilities
    "ControlSystemFactory",
    "ControlSystemConfig", 
    "ControlSystem",
    "get_factory",
    "create_neumann_voss_system",
    "create_ce_system",
    "create_neural_system",
]
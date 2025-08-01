"""
Minimal Control Theory Framework for Optimal Execution

A focused implementation for optimal execution with regime uncertainty,
based on "Optimal adaptive control with separable drift uncertainty".

Main Components:
- OptimalExecutionConfig: Problem configuration
- OptimalExecutionEnv: Environment with regime dynamics  
- Policies: REINFORCE, CertaintyEquivalent, Oracle policies
- PaperMethodsComparator: Three-method comparison tool

Usage:
    from control_theory import (
        OptimalExecutionConfig, 
        PaperMethodsComparator,
        REINFORCEConfig
    )
    
    # Setup
    config = OptimalExecutionConfig()
    reinforce_config = REINFORCEConfig(n_episodes=1000)
    
    # Comparison
    comparator = PaperMethodsComparator(config, reinforce_config)
    results = comparator.compare_all_methods(key=random.PRNGKey(42))
"""

# Configuration
from .config import OptimalExecutionConfig, default_config

# Environment
from .environment import OptimalExecutionEnv, StepResult

# Policies
from .policies import (
    Policy,
    CertaintyEquivalentPolicy,
    NaivePolicy, 
    OraclePolicy,
    RLPolicy,
    SimpleGaussianPolicy,
    SimpleDeterministicPolicy,
    create_gaussian_rl_policy,
    create_deterministic_rl_policy
)

# HJB Optimal Control (available but not used in comparison)
# from .hjb_solver import (
#     HJBConfig,
#     HJBOptimalPolicy,
#     HJBSolver,
#     ValueNetwork,
#     train_hjb_optimal_control
# )

# REINFORCE Agent
from .reinforce_agent import (
    REINFORCEConfig,
    REINFORCEPolicy,
    REINFORCEAgent,
    PolicyNetwork,
    train_reinforce_policy
)

# Comparison utilities
from .comparison import PolicyComparator, PolicyResult, PaperMethodsComparator

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "OptimalExecutionConfig",
    "default_config",
    
    # Environment
    "OptimalExecutionEnv", 
    "StepResult",
    
    # Policies
    "Policy",
    "CertaintyEquivalentPolicy",
    "NaivePolicy",
    "OraclePolicy", 
    "RLPolicy",
    "SimpleGaussianPolicy",
    "SimpleDeterministicPolicy",
    "create_gaussian_rl_policy",
    "create_deterministic_rl_policy",
    
    # HJB Optimal Control (commented out)
    # "HJBConfig",
    # "HJBOptimalPolicy", 
    # "HJBSolver",
    # "ValueNetwork",
    # "train_hjb_optimal_control",
    
    # REINFORCE Agent
    "REINFORCEConfig",
    "REINFORCEPolicy",
    "REINFORCEAgent", 
    "PolicyNetwork",
    "train_reinforce_policy",
    
    # Comparison
    "PolicyComparator",
    "PolicyResult",
    "PaperMethodsComparator",
]
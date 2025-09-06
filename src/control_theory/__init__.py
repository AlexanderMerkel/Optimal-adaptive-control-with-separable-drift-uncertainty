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


    config = OptimalExecutionConfig()
    reinforce_config = REINFORCEConfig(n_episodes=1000)


    comparator = PaperMethodsComparator(config, reinforce_config)
    results = comparator.compare_all_methods(key=random.PRNGKey(42))
"""

from .config import OptimalExecutionConfig, default_config


from .environment import OptimalExecutionEnv, StepResult


from .policies import (
    Policy,
    CertaintyEquivalentPolicy,
    NaivePolicy,
    OraclePolicy,
    RLPolicy,
    SimpleGaussianPolicy,
    SimpleDeterministicPolicy,
    create_gaussian_rl_policy,
    create_deterministic_rl_policy,
)


from .reinforce_agent import (
    REINFORCEConfig,
    REINFORCEPolicy,
    REINFORCEAgent,
    PolicyNetwork,
    train_reinforce_policy,
)


from .comparison import PolicyComparator, PolicyResult, PaperMethodsComparator


from .riccati_policies import (
    RiccatiSolver,
    RiccatiOptimalPolicy,
    RiccatiCertaintyEquivalentPolicy,
    RiccatiMeanPolicy,
)

__version__ = "1.0.0"

__all__ = [
    "OptimalExecutionConfig",
    "default_config",
    "OptimalExecutionEnv",
    "StepResult",
    "Policy",
    "CertaintyEquivalentPolicy",
    "NaivePolicy",
    "OraclePolicy",
    "RLPolicy",
    "SimpleGaussianPolicy",
    "SimpleDeterministicPolicy",
    "create_gaussian_rl_policy",
    "create_deterministic_rl_policy",
    "REINFORCEConfig",
    "REINFORCEPolicy",
    "REINFORCEAgent",
    "PolicyNetwork",
    "train_reinforce_policy",
    "PolicyComparator",
    "PolicyResult",
    "PaperMethodsComparator",
    "RiccatiSolver",
    "RiccatiOptimalPolicy",
    "RiccatiCertaintyEquivalentPolicy",
    "RiccatiMeanPolicy",
]

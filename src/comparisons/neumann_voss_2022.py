"""
Neumann-Voß 2022 Optimal Execution Controller

This implements the optimal execution solution from Neumann-Voß 2022 using mean
regime parameters. This approach treats regime uncertainty by using the population
average of regime parameters, providing a robust baseline for comparison against
adaptive methods.

Reference:
Neumann, D. and Voß, M. (2022). "Optimal execution with transient impact and
regime uncertainty." Quantitative Finance.

Mathematical Foundation:
- Uses mean regime parameters: λ_mean = 0.5*(λ_L + λ_H), κ_mean = 0.5*(κ_L + κ_H)
- Solves differential Riccati equation: dK/dt = -C_RUNNING + (λ_mean^2 / ρ^2) * K^2
- Control law: u(t) = -(λ_mean / ρ) * K(t) * X(t)
"""

import numpy as np
from jax import random
import jax.numpy as jnp

from .base_controller import BaseOptimalExecutionController
from ..utils import get_config, RiccatiSolver
from ..control_theory import RiccatiPolicy, OptimalExecutionEnvironment, TrajectoryGenerator, generate_trajectory_from_policy


class NeumannVoss2022Controller(BaseOptimalExecutionController):
    """Neumann-Voß 2022 optimal execution controller using mean regime parameters.
    
    This version uses the new control theory framework with:
    - RiccatiPolicy for optimal linear feedback control
    - OptimalExecutionEnvironment for unified environment interface
    - TrajectoryGenerator for efficient trajectory generation
    """

    def __init__(self, config=None, use_new_framework=True):
        """Initialize with centralized configuration following Neumann-Voß 2022 methodology."""
        # Initialize parent class to get state_manager, wonham_filter, etc.
        super().__init__(config)
        
        self.config = config if config is not None else get_config()
        self._init_from_config(self.config)
        
        # Initialize the original components for backward compatibility
        self.riccati_solver = RiccatiSolver(self.config)

        self.lambda_mean = 0.5 * (self.config.LAMBDA_L + self.config.LAMBDA_H)
        self.kappa_mean = 0.5 * (self.config.KAPPA_L + self.config.KAPPA_H)

        print("Neumann-Voß 2022 Controller initialized:")
        print(
            f"  λ_mean = {self.lambda_mean:.4f} (from λ_L={self.config.LAMBDA_L}, λ_H={self.config.LAMBDA_H})"
        )
        print(
            f"  κ_mean = {self.kappa_mean:.4f} (from κ_L={self.config.KAPPA_L}, κ_H={self.config.KAPPA_H})"
        )

        self.K_trajectory = self.riccati_solver.solve(self.lambda_mean)

        if np.any(np.isnan(self.K_trajectory)) or np.any(np.isinf(self.K_trajectory)):
            raise ValueError("Riccati solution contains NaN or Inf values")

        print(
            f"  Riccati solution computed successfully (K(0)={self.K_trajectory[0]:.4f}, K(T)={self.K_trajectory[-1]:.4f})"
        )
        
        # Initialize new control theory framework components if requested
        self.use_new_framework = use_new_framework
        if use_new_framework:
            self._init_control_theory_framework()
            print("  New control theory framework components initialized")

    def _init_from_config(self, config):
        """Initialize controller parameters from configuration."""
        self.T = config.T
        self.N = config.N
        self.dt = config.dt
        self.LAMBDA_L = config.LAMBDA_L
        self.LAMBDA_H = config.LAMBDA_H
        self.KAPPA_L = config.KAPPA_L
        self.KAPPA_H = config.KAPPA_H
        self.RHO = config.RHO
        self.C_RUNNING = config.C_RUNNING
        self.C_TERMINAL = config.C_TERMINAL
        self.SIGMA = config.SIGMA
        self.low_bounds = config.low_bounds
        self.high_bounds = config.high_bounds
    
    def _init_control_theory_framework(self):
        """Initialize new control theory framework components."""
        # Create RiccatiPolicy using mean lambda parameter
        self.riccati_policy = RiccatiPolicy(
            riccati_solver=self.riccati_solver,
            lambda_func=self.lambda_mean,  # Constant lambda for Neumann-Voß 2022
            rho=self.config.RHO,
            state_indices={'X': 2}  # Inventory is at index 2
        )
        
        # Create optimal execution environment
        self.control_environment = OptimalExecutionEnvironment(self.config)
        
        # Create trajectory generator for efficient rollouts
        self.trajectory_generator = TrajectoryGenerator(
            policy=self.riccati_policy,
            environment=self.control_environment,
            compile_trajectory_gen=True
        )

    def compute_control_action(self, observable_state, time_step=None):
        """Compute optimal control action using Neumann-Voß 2022 methodology."""
        X = observable_state[:, 2]
        K_current = self.K_trajectory[time_step]

        # Neumann-Voß 2022 control law: u = -(λ_mean / ρ) * K(t) * X(t)
        actions = -(self.lambda_mean / self.config.RHO) * K_current * X
        return actions

    def evaluate_performance(self, key, num_trajectories=100, n_steps=200):
        """Evaluate Neumann-Voß 2022 controller performance.
        
        Can use either the original framework or new control theory framework.
        """
        if self.use_new_framework:
            return self._evaluate_performance_new_framework(key, num_trajectories, n_steps)
        else:
            return self._evaluate_performance_original_framework(key, num_trajectories, n_steps)
    
    def _evaluate_performance_original_framework(self, key, num_trajectories=100, n_steps=200):
        """Original evaluation method for backward compatibility."""
        key, reset_key, rollout_key = random.split(key, 3)
        initial_internal, _ = self.reset_env_with_true_regime(reset_key, num_trajectories)
        states, actions, rewards, final_internal, _ = self._rollout(
            rollout_key, initial_internal, n_steps, num_trajectories
        )

        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        true_regimes_np = np.array(final_internal[:, 6])

        total_profits = np.sum(rewards_np, axis=0)
        beliefs = states_np[:, :, 3]  # Belief state p(t)
        final_beliefs = beliefs[-1, :]

        # Regime detection accuracy (based on final beliefs)
        predicted_regimes = (final_beliefs < 0.5).astype(int)
        true_regimes_int = true_regimes_np.astype(int)
        accuracy = np.mean(predicted_regimes == true_regimes_int)

        results = {
            "method": "Neumann-Voß 2022",
            "total_profits": total_profits,
            "mean_profit": np.mean(total_profits),
            "std_profit": np.std(total_profits),
            "regime_accuracy": accuracy,
            "states": states_np,
            "actions": actions_np,
            "rewards": rewards_np,
            "true_regimes": true_regimes_np,
            "beliefs": beliefs,
            "lambda_mean": float(self.lambda_mean),
            "kappa_mean": float(self.kappa_mean),
            "framework": "original"
        }

        return results
    
    def _evaluate_performance_new_framework(self, key, num_trajectories=100, n_steps=200):
        """New evaluation method using control theory framework."""
        # Generate batch trajectories using new framework
        batch_trajectories = self.trajectory_generator.generate_batch_trajectories(
            batch_size=num_trajectories,
            n_steps=n_steps,
            key=key
        )
        
        # Extract data in format compatible with original interface
        states_np = np.array(batch_trajectories.states.transpose(1, 0, 2))  # (time, batch, state_dim)
        actions_np = np.array(batch_trajectories.actions.transpose(1, 0, 1))  # (time, batch, action_dim)
        rewards_np = np.array(batch_trajectories.rewards.T)  # (time, batch)
        
        # Extract regime information from trajectory infos
        true_regimes_np = np.array(batch_trajectories.infos['true_regime'][:, -1])  # Final regime for each trajectory
        
        total_profits = np.sum(rewards_np, axis=0)
        beliefs = states_np[:, :, 3]  # Belief state p(t)
        final_beliefs = beliefs[-1, :]

        # Regime detection accuracy (based on final beliefs) 
        predicted_regimes = (final_beliefs < 0.5).astype(int)
        true_regimes_int = true_regimes_np.astype(int)
        accuracy = np.mean(predicted_regimes == true_regimes_int)

        results = {
            "method": "Neumann-Voß 2022 (Control Theory Framework)",
            "total_profits": total_profits,
            "mean_profit": np.mean(total_profits),
            "std_profit": np.std(total_profits),
            "regime_accuracy": accuracy,
            "states": states_np,
            "actions": actions_np,
            "rewards": rewards_np,
            "true_regimes": true_regimes_np,
            "beliefs": beliefs,
            "lambda_mean": float(self.lambda_mean),
            "kappa_mean": float(self.kappa_mean),
            "framework": "control_theory",
            "batch_trajectories": batch_trajectories  # Include raw trajectory data
        }

        return results

    def get_controller_info(self):
        base_info = {
            "controller_type": "Neumann-Voß 2022",
            "lambda_mean": float(self.lambda_mean),
            "kappa_mean": float(self.kappa_mean),
            "lambda_L": float(self.config.LAMBDA_L),
            "lambda_H": float(self.config.LAMBDA_H),
            "kappa_L": float(self.config.KAPPA_L),
            "kappa_H": float(self.config.KAPPA_H),
            "riccati_solver": "Shared RiccatiSolver utility (DOP853/Radau)",
            "K_initial": float(self.K_trajectory[0]),
            "K_terminal": float(self.K_trajectory[-1]),
            "framework": "hybrid" if self.use_new_framework else "original"
        }
        
        if self.use_new_framework:
            base_info.update({
                "riccati_policy": "RiccatiPolicy with constant lambda_mean",
                "environment": "OptimalExecutionEnvironment",
                "trajectory_generator": "TrajectoryGenerator (JAX compiled)",
                "policy_type": "deterministic",
                "time_varying": True
            })
        
        return base_info


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Neumann-Voß 2022 Optimal Execution Controller")
    print("=" * 60)

    key = random.PRNGKey(42)
    
    # Test original framework
    print("\n" + "=" * 60)
    print("Testing Original Framework")
    print("=" * 60)
    
    controller_original = NeumannVoss2022Controller(use_new_framework=False)
    info_original = controller_original.get_controller_info()
    print("\nOriginal Framework Configuration:")
    for key_name, value in info_original.items():
        print(f"  {key_name}: {value}")

    print("\nEvaluating original framework performance...")
    key, eval_key = random.split(key)
    results_original = controller_original.evaluate_performance(eval_key, num_trajectories=50, n_steps=100)

    print("\nOriginal Framework Results:")
    print(f"  Mean profit: {results_original['mean_profit']:.4f} ± {results_original['std_profit']:.4f}")
    print(f"  Regime detection accuracy: {results_original['regime_accuracy']:.1%}")
    print(f"  Profit range: [{np.min(results_original['total_profits']):.2f}, {np.max(results_original['total_profits']):.2f}]")
    print(f"  Numerical stability: {'✓' if not np.any(np.isnan(results_original['total_profits'])) else '✗'}")
    
    # Test new control theory framework
    print("\n" + "=" * 60)
    print("Testing New Control Theory Framework")
    print("=" * 60)
    
    controller_new = NeumannVoss2022Controller(use_new_framework=True)
    info_new = controller_new.get_controller_info()
    print("\nNew Framework Configuration:")
    for key_name, value in info_new.items():
        print(f"  {key_name}: {value}")

    print("\nEvaluating new framework performance...")
    key, eval_key = random.split(key)
    results_new = controller_new.evaluate_performance(eval_key, num_trajectories=50, n_steps=100)

    print("\nNew Framework Results:")
    print(f"  Mean profit: {results_new['mean_profit']:.4f} ± {results_new['std_profit']:.4f}")
    print(f"  Regime detection accuracy: {results_new['regime_accuracy']:.1%}")
    print(f"  Profit range: [{np.min(results_new['total_profits']):.2f}, {np.max(results_new['total_profits']):.2f}]")
    print(f"  Numerical stability: {'✓' if not np.any(np.isnan(results_new['total_profits'])) else '✗'}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("Framework Comparison")
    print("=" * 60)
    print(f"Original framework mean profit: {results_original['mean_profit']:.4f}")
    print(f"New framework mean profit:      {results_new['mean_profit']:.4f}")
    print(f"Difference:                     {abs(results_new['mean_profit'] - results_original['mean_profit']):.4f}")
    print(f"Relative difference:            {abs(results_new['mean_profit'] - results_original['mean_profit']) / abs(results_original['mean_profit']) * 100:.2f}%")
    
    print(f"\nOriginal framework accuracy: {results_original['regime_accuracy']:.1%}")
    print(f"New framework accuracy:      {results_new['regime_accuracy']:.1%}")
    
    # Verify numerical equivalence (should be very close)
    if abs(results_new['mean_profit'] - results_original['mean_profit']) < 0.01:
        print("\n✓ Frameworks produce numerically equivalent results")
    else:
        print("\n⚠ Frameworks show significant differences - needs investigation")

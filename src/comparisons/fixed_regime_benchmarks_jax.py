"""
Fixed Regime Parameter Benchmark for Optimal Execution

This implements a benchmark solution using mean regime parameters:
- Mixed Baseline: Uses population average λ = 0.5*(λ_L + λ_H), κ = 0.5*(κ_L + κ_H)

This provides a baseline for comparing against adaptive control methods and forms
the basis for the Neumann-Voß 2022 optimal execution solution.
"""

# Third-party imports
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.integrate import solve_ivp

# Local imports
from .base_controller import BaseOptimalExecutionController


class FixedRegimeBenchmarks(BaseOptimalExecutionController):
    """Fixed regime parameter benchmark using mean parameters or specific regime parameters."""

    def __init__(self, config=None, regime_type="mean"):
        """Initialize with model parameters.

        Args:
            config: Configuration object, if None uses default config
            regime_type: 'mean', 'low', or 'high' regime parameters
        """
        super().__init__(config)

        self.regime_type = regime_type

        if regime_type == "mean":
            # Compute mean regime parameters (basis for Neumann-Voß 2022)
            self.lambda_val = 0.5 * (self.LAMBDA_L + self.LAMBDA_H)
            self.kappa_val = 0.5 * (self.KAPPA_L + self.KAPPA_H)
            self.method_name = "Fixed Regime (Mean Parameters)"
        elif regime_type == "low":
            # Use low regime parameters
            self.lambda_val = self.LAMBDA_L
            self.kappa_val = self.KAPPA_L
            self.method_name = "Fixed Regime (Low Parameters)"
        elif regime_type == "high":
            # Use high regime parameters
            self.lambda_val = self.LAMBDA_H
            self.kappa_val = self.KAPPA_H
            self.method_name = "Fixed Regime (High Parameters)"
        else:
            raise ValueError(f"Invalid regime_type: {regime_type}. Must be 'mean', 'low', or 'high'")

        # Pre-compute Riccati solution for selected parameters
        self.K_trajectory = self._solve_riccati_equation(self.lambda_val)

    def _solve_riccati_equation(self, lambda_val):
        """Solve the Riccati equation using SciPy's robust ODE solver."""
        # For the LQR problem with transient impact:
        # dK/dt = -C_RUNNING + (lambda_val^2 / RHO^2) * K^2
        # Terminal condition: K(T) = C_TERMINAL

        # Coefficients for the Riccati equation
        a = lambda_val**2 / (self.RHO**2)
        c = self.C_RUNNING

        def riccati_ode(t, K):
            """ODE function for the differential Riccati equation."""
            # dK/dt = -c + a*K^2 (note: we integrate backward, so negate time derivative)
            return -c + a * K[0]**2

        # Time span for backward integration (from T to 0)
        t_span = (self.T, 0.0)
        t_eval = np.linspace(self.T, 0.0, self.N + 1)

        # Initial condition (at terminal time T)
        K0 = [self.C_TERMINAL]

        # Solve using SciPy's robust ODE solver
        sol = solve_ivp(
            riccati_ode,
            t_span,
            K0,
            t_eval=t_eval,
            method='DOP853',  # High-order Runge-Kutta method
            rtol=1e-8,        # Relative tolerance
            atol=1e-10        # Absolute tolerance
        )

        if not sol.success:
            print(f"Warning: ODE solver failed for lambda={lambda_val}: {sol.message}")
            # Fallback to simpler method if needed
            sol = solve_ivp(
                riccati_ode,
                t_span,
                K0,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )

        K_solution = sol.y[0, :][::-1]  # Reverse for forward time

        # Convert to JAX array for compatibility
        K = jnp.array(K_solution, dtype=jnp.float32)

        return K

    def compute_control_action(self, observable_state, time_step=None):
        """Compute control action using fixed regime parameters."""
        t, S, X, p, A_l, A_h = observable_state.T

        K_values = self.K_trajectory[time_step]

        # Fixed regime control law: u = -(lambda_val / rho) * K * X
        actions = -(self.lambda_val / self.RHO) * K_values * X

        return actions

    def evaluate_performance(self, key, num_trajectories=100, n_steps=200):
        """Evaluate performance using fixed regime parameters."""
        key, reset_key, rollout_key = random.split(key, 3)

        initial_internal, _ = self.reset_env_with_true_regime(reset_key, num_trajectories)

        states, actions, rewards, final_internal, infos = self._rollout(
            rollout_key, initial_internal, n_steps, num_trajectories
        )

        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        true_regimes_np = np.array(final_internal[:, 6])

        # Performance metrics
        total_profits = np.sum(rewards_np, axis=0)
        beliefs = states_np[:, :, 3]  # p(t) (not used in control but still evolves)
        final_beliefs = beliefs[-1, :]

        # Regime detection accuracy (based on belief evolution, not control)
        predicted_regimes = (final_beliefs < 0.5).astype(int)
        true_regimes_int = true_regimes_np.astype(int)
        accuracy = np.mean(predicted_regimes == true_regimes_int)

        results = {
            'method': self.method_name,
            'total_profits': total_profits,
            'mean_profit': np.mean(total_profits),
            'std_profit': np.std(total_profits),
            'regime_accuracy': accuracy,
            'states': states_np,
            'actions': actions_np,
            'rewards': rewards_np,
            'true_regimes': true_regimes_np,
            'beliefs': beliefs
        }

        return results



if __name__ == "__main__":
    # Test the fixed regime benchmark with mean parameters
    key = random.PRNGKey(42)
    benchmarks = FixedRegimeBenchmarks()

    print("Testing Fixed Regime Benchmark (Mean Parameters)...")
    results = benchmarks.evaluate_performance(key, num_trajectories=50)

    print(f"\n{results['method']}:")
    print(f"  Mean profit: {results['mean_profit']:.4f}")
    print(f"  Profit std: {results['std_profit']:.4f}")
    print(f"  Regime detection accuracy: {results['regime_accuracy']:.1%}")
    print(f"  Lambda val: {benchmarks.lambda_val:.4f}")
    print(f"  Kappa val: {benchmarks.kappa_val:.4f}")
    print(f"  Regime type: {benchmarks.regime_type}")

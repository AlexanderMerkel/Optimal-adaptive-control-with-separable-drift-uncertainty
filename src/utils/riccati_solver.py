"""
Riccati Equation Solver for Optimal Execution Controllers

This module provides a unified implementation of the Riccati equation solver
that was previously duplicated across multiple controllers. It consolidates
the ODE solving logic and provides both single-solution and grid-based solving
capabilities with robust error handling.

Mathematical Foundation:
    For the LQR problem with transient impact:
    dK/dt = -C_RUNNING + (lambda^2 / RHO^2) * K^2
    Terminal condition: K(T) = C_TERMINAL

Usage:
    from utils.riccati_solver import RiccatiSolver
    from utils.config import get_config

    config = get_config()
    solver = RiccatiSolver(config)

    # Single solution
    K_trajectory = solver.solve(lambda_val=1.0)

    # Grid of solutions
    lambda_grid = jnp.linspace(0.5, 1.5, 101)
    K_solutions = solver.solve_grid(lambda_grid)
"""

from typing import Union, Optional, Tuple
import warnings

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from .config import Config


class RiccatiSolver:
    """Unified Riccati equation solver for optimal execution controllers."""

    def __init__(self, config: Config):
        """Initialize solver with configuration parameters.

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.T = config.T
        self.N = config.N
        self.time_grid = config.time_grid
        self.C_RUNNING = config.C_RUNNING
        self.C_TERMINAL = config.C_TERMINAL
        self.RHO = config.RHO

        # Solver configuration
        self.default_method = 'DOP853'  # High-order Runge-Kutta
        self.fallback_method = 'Radau'  # Implicit method for stiff equations
        self.default_rtol = 1e-8
        self.default_atol = 1e-10
        self.fallback_rtol = 1e-8
        self.fallback_atol = 1e-10

    def solve(
        self,
        lambda_val: float,
        method: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        validate: bool = True
    ) -> jnp.ndarray:
        """Solve Riccati equation for a single lambda value.

        Args:
            lambda_val: Regime parameter value
            method: ODE solver method ('DOP853', 'Radau', etc.)
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            validate: Whether to validate solution for NaN/Inf

        Returns:
            K_trajectory: Array of K values from t=0 to t=T

        Raises:
            ValueError: If solution contains NaN or Inf values
            RuntimeError: If ODE solver fails with fallback
        """
        method = method or self.default_method
        rtol = rtol or self.default_rtol
        atol = atol or self.default_atol

        try:
            sol = self._solve_ode(lambda_val, method, rtol, atol)

            if not sol.success:
                warnings.warn(f"Primary ODE solver failed for λ={lambda_val}: {sol.message}")
                sol = self._solve_ode(lambda_val, self.fallback_method, self.fallback_rtol, self.fallback_atol)

                if not sol.success:
                    raise RuntimeError(f"All ODE solvers failed for λ={lambda_val}: {sol.message}")

            K_solution = sol.y[0, :][::-1]  # Reverse for forward time order
            K_trajectory = jnp.array(K_solution, dtype=jnp.float32)

            if validate:
                self._validate_solution(K_trajectory, lambda_val)

            return K_trajectory

        except Exception as e:
            raise RuntimeError(f"Riccati solver failed for λ={lambda_val}: {e}")

    def solve_grid(
        self,
        lambda_grid: Union[jnp.ndarray, np.ndarray],
        method: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        validate: bool = True
    ) -> jnp.ndarray:
        """Solve Riccati equation for multiple lambda values.

        Args:
            lambda_grid: Array of lambda values to solve for
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            validate: Whether to validate solutions

        Returns:
            K_solutions: Array of shape (len(lambda_grid), N+1) containing solutions
        """
        solutions = []

        for lam in lambda_grid:
            K_trajectory = self.solve(
                float(lam), method=method, rtol=rtol, atol=atol, validate=validate
            )
            solutions.append(K_trajectory)

        return jnp.array(solutions)

    def _solve_ode(
        self,
        lambda_val: float,
        method: str,
        rtol: float,
        atol: float
    ) -> object:
        """Internal ODE solving method.

        Args:
            lambda_val: Regime parameter value
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Solution object from scipy.integrate.solve_ivp
        """
        # Coefficients for the Riccati equation
        a = lambda_val**2 / (self.RHO**2)
        c = self.C_RUNNING

        def riccati_ode(t, K):
            """ODE function for the differential Riccati equation.

            The equation is: dK/dt = -c + a*K^2
            We integrate backward in time, so we negate the time derivative.
            """
            return -c + a * K[0]**2

        t_span = (self.T, 0.0)  # Backward integration from T to 0
        t_eval = np.linspace(self.T, 0.0, self.N + 1)
        K0 = [self.C_TERMINAL]  # Terminal condition at T

        return solve_ivp(
            riccati_ode,
            t_span,
            K0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol
        )

    def _validate_solution(self, K_trajectory: jnp.ndarray, lambda_val: float):
        """Validate Riccati solution for numerical issues.

        Args:
            K_trajectory: Solution trajectory to validate
            lambda_val: Lambda value used for context in error messages

        Raises:
            ValueError: If solution contains NaN, Inf, or negative values
        """
        if jnp.any(jnp.isnan(K_trajectory)):
            raise ValueError(f"Riccati solution contains NaN values for λ={lambda_val}")

        if jnp.any(jnp.isinf(K_trajectory)):
            raise ValueError(f"Riccati solution contains Inf values for λ={lambda_val}")

        if jnp.any(K_trajectory < 0):
            warnings.warn(f"Riccati solution contains negative values for λ={lambda_val}")

    def interpolate_solution(
        self,
        K_solutions: jnp.ndarray,
        lambda_grid: jnp.ndarray,
        lambda_val: float,
        time_idx: int
    ) -> float:
        """Interpolate Riccati solution for arbitrary lambda and time.

        Args:
            K_solutions: Pre-computed solutions grid (lambda_grid_size, N+1)
            lambda_grid: Lambda values corresponding to solutions
            lambda_val: Target lambda value for interpolation
            time_idx: Time index for interpolation

        Returns:
            Interpolated K value
        """
        K_at_time = K_solutions[:, time_idx]
        return jnp.interp(lambda_val, lambda_grid, K_at_time)

    def get_analytical_bounds(self, lambda_min: float, lambda_max: float) -> Tuple[float, float]:
        """Get analytical bounds for Riccati solutions.

        Args:
            lambda_min: Minimum lambda value
            lambda_max: Maximum lambda value

        Returns:
            Tuple of (K_min, K_max) bounds
        """
        # For the given Riccati equation, K is bounded by terminal condition
        # and steady-state solution: K_ss = sqrt(C_RUNNING) * RHO / lambda
        K_terminal = self.C_TERMINAL
        K_ss_min = jnp.sqrt(self.C_RUNNING) * self.RHO / lambda_max
        K_ss_max = jnp.sqrt(self.C_RUNNING) * self.RHO / lambda_min
        K_min = min(float(K_terminal), float(K_ss_min))
        K_max = max(float(K_terminal), float(K_ss_max))

        return K_min, K_max

    def print_solver_info(self):
        """Print solver configuration information."""
        print("=== Riccati Solver Configuration ===")
        print(f"Time horizon: T = {self.T}")
        print(f"Time steps: N = {self.N}")
        print(f"Cost parameters: C_running = {self.C_RUNNING}, C_terminal = {self.C_TERMINAL}")
        print(f"Temporary impact: RHO = {self.RHO}")
        print(f"Default method: {self.default_method} (fallback: {self.fallback_method})")
        print(f"Default tolerances: rtol = {self.default_rtol}, atol = {self.default_atol}")
        print("=" * 40)


def create_riccati_solver(config: Optional[Config] = None) -> RiccatiSolver:
    """Factory function to create Riccati solver with configuration.

    Args:
        config: Configuration object, uses default if None

    Returns:
        Configured RiccatiSolver instance
    """
    if config is None:
        from .config import get_config
        config = get_config()

    return RiccatiSolver(config)


def solve_single_riccati(lambda_val: float, config: Optional[Config] = None) -> jnp.ndarray:
    """Solve Riccati equation for a single lambda value.

    Args:
        lambda_val: Regime parameter value
        config: Configuration object, uses default if None

    Returns:
        K_trajectory: Solution trajectory
    """
    solver = create_riccati_solver(config)
    return solver.solve(lambda_val)


def solve_riccati_grid(lambda_grid: jnp.ndarray, config: Optional[Config] = None) -> jnp.ndarray:
    """Solve Riccati equation for grid of lambda values.

    Args:
        lambda_grid: Array of lambda values
        config: Configuration object, uses default if None

    Returns:
        K_solutions: Grid of solutions
    """
    solver = create_riccati_solver(config)
    return solver.solve_grid(lambda_grid)


if __name__ == "__main__":
    # Test the Riccati solver
    from .config import get_config

    config = get_config()
    solver = RiccatiSolver(config)
    solver.print_solver_info()

    # Test single solution
    print("Testing single solution...")
    K_traj = solver.solve(lambda_val=1.0)
    print(f"Solution range: K(0) = {K_traj[0]:.4f}, K(T) = {K_traj[-1]:.4f}")

    # Test grid solution
    print("Testing grid solution...")
    lambda_grid = jnp.linspace(0.5, 1.5, 5)
    K_solutions = solver.solve_grid(lambda_grid)
    print(f"Grid solutions shape: {K_solutions.shape}")

    # Test interpolation
    K_interp = solver.interpolate_solution(K_solutions, lambda_grid, 1.0, 0)
    print(f"Interpolated K(0) for λ=1.0: {K_interp:.4f}")

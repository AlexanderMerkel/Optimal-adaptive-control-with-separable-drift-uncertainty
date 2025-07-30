"""
Control Environment Abstractions

This module provides abstractions for control environments that encapsulate
the system dynamics, reward computation, and state management for control problems.

The environment abstraction separates the control policy from the system dynamics,
enabling reusable components and easier testing.

Key Components:
    - ControlEnvironment: Abstract base class for control environments
    - OptimalExecutionEnvironment: Specific implementation for optimal execution
    - StateSpace: Definition of valid state space with bounds
    - NoiseModel: Abstraction for different noise processes
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Union
import jax
import jax.numpy as jnp
from jax import random

from .core import State, Action, Reward, Info, StateTransitionSystem, RewardFunction, create_state
from ..utils import Config, WonhamFilter, StateManager, PriceDynamics


class StateSpace:
    """
    Definition of the state space with bounds and constraints.
    
    Provides utilities for:
    - State validation and clipping
    - Random state sampling  
    - State space visualization
    """
    
    def __init__(self, 
                 bounds: Dict[str, Tuple[float, float]],
                 state_names: list,
                 initial_state: Dict[str, float]):
        """
        Initialize state space.
        
        Args:
            bounds: Dictionary mapping state variable names to (min, max) bounds
            state_names: Ordered list of state variable names
            initial_state: Default initial state values
        """
        self.bounds = bounds
        self.state_names = state_names
        self.initial_state = initial_state
        
        # Create bounds arrays for efficient clipping
        self.low_bounds = jnp.array([bounds[name][0] for name in state_names])
        self.high_bounds = jnp.array([bounds[name][1] for name in state_names])
        
        # Create initial state array
        self.initial_state_array = jnp.array([initial_state[name] for name in state_names])
    
    def clip_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Clip state to valid bounds."""
        return jnp.clip(state, self.low_bounds, self.high_bounds)
    
    def sample_initial_state(self, key: random.PRNGKey, batch_size: int = 1) -> jnp.ndarray:
        """Sample random initial states within bounds."""
        if batch_size == 1:
            return self.initial_state_array
        else:
            return jnp.tile(self.initial_state_array, (batch_size, 1))
    
    def is_valid_state(self, state: jnp.ndarray) -> bool:
        """Check if state is within valid bounds."""
        return jnp.all((state >= self.low_bounds) & (state <= self.high_bounds))
    
    @property
    def dimension(self) -> int:
        """Dimension of the state space."""
        return len(self.state_names)


class NoiseModel(ABC):
    """
    Abstract base class for noise models in stochastic control.
    
    Different noise models:
    - Brownian motion (Wiener process)
    - Jump processes (Poisson jumps)
    - Regime switching noise
    - Correlated noise processes
    """
    
    @abstractmethod
    def sample(self, key: random.PRNGKey, shape: tuple, dt: float) -> jnp.ndarray:
        """
        Sample noise realization.
        
        Args:
            key: Random key
            shape: Shape of noise sample
            dt: Time step
            
        Returns:
            Noise realization
        """
        pass


class BrownianMotion(NoiseModel):
    """Standard Brownian motion noise model."""
    
    def __init__(self, volatility: float = 1.0):
        """
        Initialize Brownian motion.
        
        Args:
            volatility: Volatility parameter σ
        """
        self.volatility = volatility
    
    def sample(self, key: random.PRNGKey, shape: tuple, dt: float) -> jnp.ndarray:
        """Sample Brownian motion increment."""
        return self.volatility * jnp.sqrt(dt) * random.normal(key, shape)


class ControlEnvironment(ABC):
    """
    Abstract base class for control environments.
    
    A control environment encapsulates:
    - System dynamics (state transitions)
    - Reward/cost computation
    - State space definition and constraints
    - Noise processes
    - Terminal conditions
    
    This abstraction enables:
    - Separation of policy and environment
    - Reusable environment components
    - Easier testing and validation
    - Standardized interfaces across different control problems
    """
    
    def __init__(self, 
                 state_space: StateSpace,
                 dynamics: StateTransitionSystem,
                 reward_function: RewardFunction,
                 noise_model: NoiseModel):
        """
        Initialize control environment.
        
        Args:
            state_space: State space definition
            dynamics: System dynamics
            reward_function: Reward/cost function
            noise_model: Noise process model
        """
        self.state_space = state_space
        self.dynamics = dynamics
        self.reward_function = reward_function
        self.noise_model = noise_model
    
    @abstractmethod
    def reset(self, key: random.PRNGKey, batch_size: int = 1) -> State:
        """
        Reset environment to initial state.
        
        Args:
            key: Random key for initialization
            batch_size: Number of parallel environments
            
        Returns:
            Initial state
        """
        pass
    
    @abstractmethod
    def step(self, key: random.PRNGKey, state: State, action: Action) -> Tuple[State, Reward, bool, Info]:
        """
        Execute one environment step.
        
        Args:
            key: Random key for noise generation
            state: Current state
            action: Applied action
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
    
    def batch_step(self, key: random.PRNGKey, states: State, actions: Action) -> Tuple[State, Reward, jnp.ndarray, Info]:
        """
        Execute batch of environment steps.
        
        Args:
            key: Random key
            states: Batch of current states
            actions: Batch of actions
            
        Returns:
            Tuple of (next_states, rewards, done_flags, info)
        """
        # Default implementation using vmap
        batch_fn = jax.vmap(self.step, in_axes=(None, 0, 0))
        keys = random.split(key, states.shape[0])
        return batch_fn(keys, states, actions)
    
    @property
    def state_dim(self) -> int:
        """Dimension of state space."""
        return self.state_space.dimension
    
    @property 
    def action_dim(self) -> int:
        """Dimension of action space."""
        return self.dynamics.action_dim


class OptimalExecutionEnvironment(ControlEnvironment):
    """
    Optimal execution environment implementing the mathematical model from
    the research paper.
    
    State: [t, S, X, p, A_l, A_h] (6D observable) + regime (hidden)
    Dynamics:
        - dS = (actual_drift)dt + σdW (price dynamics)
        - dX = -u dt (inventory dynamics)  
        - dp = Wonham filter update (belief dynamics)
        - dA_l = (u + κ_L A_l)dt (low regime accumulator)
        - dA_h = (u + κ_H A_h)dt (high regime accumulator)
    
    Rewards: r(S,X,u) = [(S - ρu)u - c X²]dt + terminal reward
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize optimal execution environment.
        
        Args:
            config: Configuration object (uses default if None)
        """
        if config is None:
            from ..utils import get_config
            config = get_config()
        
        self.config = config
        
        # Create mathematical utilities
        self.wonham_filter = WonhamFilter(config)
        self.state_manager = StateManager(config)
        self.price_dynamics = PriceDynamics(config)
        
        # Create state space definition
        state_space = StateSpace(
            bounds=config.STATE_BOUNDS,
            state_names=config.STATE_NAMES,
            initial_state=config.INITIAL_STATE
        )
        
        # Create dynamics and reward components
        dynamics = OptimalExecutionDynamics(config, self.wonham_filter, self.state_manager)
        reward_function = OptimalExecutionReward(config, self.price_dynamics)
        noise_model = BrownianMotion(volatility=config.SIGMA)
        
        super().__init__(state_space, dynamics, reward_function, noise_model)
        
        # Cache frequently used parameters
        self.dt = config.dt
        self.sqrt_dt = jnp.sqrt(config.dt)
    
    def reset(self, key: random.PRNGKey, batch_size: int = 1) -> State:
        """
        Reset to initial state with random regime assignment.
        
        Args:
            key: Random key
            batch_size: Number of parallel environments
            
        Returns:
            Initial state with hidden regime
        """
        internal_states, observable_states = self.state_manager.initialize_batch(batch_size, key)
        
        # Return observable state with hidden regime in metadata
        if batch_size == 1:
            regime = internal_states[0, 6]  # Extract regime for single env
            return create_state(
                data=observable_states[0],
                time=0.0,
                regime=regime
            )
        else:
            regimes = internal_states[:, 6]  # Extract regimes for batch
            return create_state(
                data=observable_states,
                time=0.0,
                regime=regimes
            )
    
    def step(self, key: random.PRNGKey, state: State, action: Action) -> Tuple[State, Reward, bool, Info]:
        """
        Execute one step of optimal execution dynamics.
        
        Args:
            key: Random key for noise
            state: Current state (observable + metadata with regime)
            action: Control action
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Extract state components
        observable_data = state.data
        regime = state.metadata['regime'] if state.metadata else 0.0
        current_time = state.time if state.time is not None else 0.0
        
        # Reconstruct internal state (observable + regime)  
        if observable_data.ndim == 1:
            internal_state = jnp.concatenate([observable_data, jnp.array([regime])])
        else:
            # Batch case
            batch_size = observable_data.shape[0]
            regime_array = jnp.atleast_1d(regime)
            if regime_array.shape[0] == 1:
                regime_array = jnp.tile(regime_array, batch_size)
            internal_state = jnp.column_stack([observable_data, regime_array])
        
        # Generate noise
        noise_shape = (1,) if observable_data.ndim == 1 else (observable_data.shape[0],)
        dW = self.noise_model.sample(key, noise_shape, self.dt)
        
        # Apply dynamics
        next_internal = self.dynamics.transition(
            create_state(data=internal_state, time=current_time),
            action,
            dW,
            self.dt,
            current_time
        )
        
        # Extract next observable state and regime
        next_observable = next_internal.data[..., :6]  # First 6 components
        next_regime = next_internal.data[..., 6]       # Last component
        
        # Compute reward
        reward = self.reward_function.compute_reward(
            create_state(data=next_observable),
            action,
            current_time + self.dt
        )
        
        # Check terminal condition (simplified: based on time)
        next_time = current_time + self.dt
        done = next_time >= self.config.T
        
        # Create next state
        next_state = create_state(
            data=next_observable,
            time=next_time,
            regime=next_regime
        )
        
        # Create info dictionary
        info = Info(
            data={
                'true_regime': next_regime,
                'belief': next_observable[..., 3] if next_observable.ndim > 1 else next_observable[3],
                'inventory': next_observable[..., 2] if next_observable.ndim > 1 else next_observable[2],
                'price': next_observable[..., 1] if next_observable.ndim > 1 else next_observable[1]
            }
        )
        
        return next_state, reward, done, info


class OptimalExecutionDynamics(StateTransitionSystem):
    """State transition system for optimal execution problem."""
    
    def __init__(self, config: Config, wonham_filter: WonhamFilter, state_manager: StateManager):
        """Initialize dynamics with mathematical utilities."""
        self.config = config
        self.wonham_filter = wonham_filter
        self.state_manager = state_manager
    
    def transition(self, state: State, action: Action, noise: jnp.ndarray,
                  dt: float, time: Optional[float] = None) -> State:
        """Apply optimal execution state transition."""
        # Handle single trajectory case by adding batch dimension
        internal_state = state.data
        actions = action.data
        dW = noise
        
        # Add batch dimension if needed (single trajectory)
        if internal_state.ndim == 1:
            internal_state = jnp.expand_dims(internal_state, 0)
            actions = jnp.expand_dims(actions, 0) 
            dW = jnp.expand_dims(dW, 0)
            single_trajectory = True
        else:
            single_trajectory = False
            
        # Use the existing update_full_state function from math_utils
        from ..utils import update_full_state
        
        next_internal, next_observable, rewards, info = update_full_state(
            internal_state, actions, dW, self.config
        )
        
        # Remove batch dimension if we added it
        if single_trajectory:
            next_internal = next_internal[0]
        
        return create_state(data=next_internal, time=(time or 0.0) + dt)
    
    @property
    def state_dim(self) -> int:
        """State dimension (6D observable + 1D regime).""" 
        return 7
    
    @property
    def action_dim(self) -> int:
        """Action dimension (1D control)."""
        return 1


class OptimalExecutionReward(RewardFunction):
    """Reward function for optimal execution problem."""
    
    def __init__(self, config: Config, price_dynamics: PriceDynamics):
        """Initialize reward function."""
        self.config = config
        self.price_dynamics = price_dynamics
    
    def compute_reward(self, state: State, action: Action, 
                      time: Optional[float] = None) -> Reward:
        """Compute instantaneous reward for optimal execution."""
        # Extract price and inventory from state
        state_data = state.data
        if state_data.ndim == 1:
            S = state_data[1]  # Price
            X = state_data[2]  # Inventory
        else:
            S = state_data[:, 1]  # Batch price
            X = state_data[:, 2]  # Batch inventory
        
        actions = action.data
        reward_value = self.price_dynamics.compute_reward(S, X, actions)
        
        return Reward(
            value=reward_value,
            components={
                'execution_reward': (S - self.config.RHO * actions) * actions * self.config.dt,
                'inventory_cost': -self.config.C_RUNNING * X**2 * self.config.dt
            }
        )
    
    def compute_terminal_reward(self, state: State) -> Reward:
        """Compute terminal reward."""
        state_data = state.data
        if state_data.ndim == 1:
            S_final = state_data[1]
            X_final = state_data[2]
        else:
            S_final = state_data[:, 1]
            X_final = state_data[:, 2]
        
        terminal_value = self.price_dynamics.compute_terminal_reward(S_final, X_final)
        
        return Reward(
            value=terminal_value,
            components={
                'liquidation_value': S_final * X_final,
                'terminal_penalty': -self.config.C_TERMINAL * X_final**2
            }
        )
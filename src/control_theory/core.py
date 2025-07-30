"""
Core Abstractions for Control Theory Framework

This module defines the fundamental types and interfaces that form the foundation
of the control theory framework. All components build upon these abstractions.

Mathematical Foundation:
    - State space: X ⊆ ℝⁿ
    - Action space: U ⊆ ℝᵐ  
    - Dynamics: dx = f(x,u,t)dt + σ(x,u,t)dW
    - Cost: J = E[∫₀ᵀ L(x,u,t)dt + Φ(x(T))]
    - Policy: π: X × [0,T] → U
"""

from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, Any, Callable, Optional
import jax.numpy as jnp
from jax import random


class State(NamedTuple):
    """
    Immutable state representation for control problems.
    
    Provides a structured, type-safe way to represent system states with
    automatic JAX compatibility for gradient computation and JIT compilation.
    
    Attributes:
        data: JAX array containing state variables
        time: Current time (optional, can be part of data)
        metadata: Additional state information (belief states, accumulators, etc.)
    """
    data: jnp.ndarray
    time: Optional[float] = None
    metadata: Optional[Dict[str, jnp.ndarray]] = None
    
    @property
    def shape(self) -> tuple:
        """Shape of the state data array."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions in state space."""
        return self.data.shape[-1] if self.data.ndim > 0 else 0
    
    def __getitem__(self, key):
        """Allow indexing into the state data."""
        return self.data[key]


class Action(NamedTuple):
    """
    Immutable action representation for control problems.
    
    Attributes:
        data: JAX array containing control actions
        metadata: Additional action information (policy parameters, etc.)
    """
    data: jnp.ndarray
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def shape(self) -> tuple:
        """Shape of the action data array."""
        return self.data.shape
    
    def __getitem__(self, key):
        """Allow indexing into the action data."""
        return self.data[key]


class Reward(NamedTuple):
    """
    Immutable reward representation.
    
    Attributes:
        value: Scalar or array reward values
        components: Breakdown of reward components (optional)
    """
    value: jnp.ndarray
    components: Optional[Dict[str, jnp.ndarray]] = None


class Info(NamedTuple):
    """
    Additional information from environment steps.
    
    Attributes:
        data: Dictionary of additional information
        diagnostics: Diagnostic information for debugging
    """
    data: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None


class ControlPolicy(ABC):
    """
    Abstract base class for control policies.
    
    A control policy maps states and time to actions: π: X × [0,T] → U
    
    This abstraction enables:
    - Deterministic policies (LQR, MPC)
    - Stochastic policies (neural networks, randomized)
    - Time-varying policies (finite horizon problems)
    - Belief-dependent policies (partially observable problems)
    """
    
    @abstractmethod
    def compute_action(self, state: State, time: Optional[float] = None, 
                      key: Optional[random.PRNGKey] = None) -> Action:
        """
        Compute control action for given state and time.
        
        Args:
            state: Current system state
            time: Current time (if not included in state)
            key: Random key for stochastic policies
            
        Returns:
            Control action
        """
        pass
    
    def batch_compute_action(self, states: State, times: Optional[jnp.ndarray] = None,
                           key: Optional[random.PRNGKey] = None) -> Action:
        """
        Compute actions for batch of states (default implementation).
        
        Subclasses can override for more efficient batch computation.
        
        Args:
            states: Batch of states
            times: Array of times for each state
            key: Random key for stochastic policies
            
        Returns:
            Batch of actions
        """
        # Default implementation using vmap
        import jax
        
        if times is not None:
            # Vmap over states and times
            batch_fn = jax.vmap(self.compute_action, in_axes=(0, 0, None))
            return batch_fn(states, times, key)
        else:
            # Vmap over states only
            batch_fn = jax.vmap(self.compute_action, in_axes=(0, None, None))
            return batch_fn(states, time=None, key=key)
    
    @property
    def is_stochastic(self) -> bool:
        """Whether this policy is stochastic (requires random key)."""
        return False
        
    @property
    def is_time_varying(self) -> bool:
        """Whether this policy depends explicitly on time."""
        return False


class StateTransitionSystem(ABC):
    """
    Abstract base class for state transition dynamics.
    
    Represents the system dynamics: dx = f(x,u,t)dt + σ(x,u,t)dW
    
    This abstraction enables:
    - Deterministic dynamics (ODEs)
    - Stochastic dynamics (SDEs)  
    - Discrete-time systems
    - Hybrid systems
    - Jump-diffusion processes
    """
    
    @abstractmethod
    def transition(self, state: State, action: Action, noise: jnp.ndarray,
                  dt: float, time: Optional[float] = None) -> State:
        """
        Compute next state given current state, action, and noise.
        
        Args:
            state: Current state
            action: Applied action
            noise: Random noise (e.g., Brownian increment)
            dt: Time step
            time: Current time
            
        Returns:
            Next state
        """
        pass
    
    def batch_transition(self, states: State, actions: Action, noise: jnp.ndarray,
                        dt: float, times: Optional[jnp.ndarray] = None) -> State:
        """
        Compute batch of state transitions (default implementation).
        
        Args:
            states: Batch of current states  
            actions: Batch of actions
            noise: Batch of noise realizations
            dt: Time step
            times: Array of current times
            
        Returns:
            Batch of next states
        """
        import jax
        
        if times is not None:
            batch_fn = jax.vmap(self.transition, in_axes=(0, 0, 0, None, 0))
            return batch_fn(states, actions, noise, dt, times)
        else:
            batch_fn = jax.vmap(self.transition, in_axes=(0, 0, 0, None, None))
            return batch_fn(states, actions, noise, dt, time=None)
    
    @property
    def is_stochastic(self) -> bool:
        """Whether this system has stochastic dynamics."""
        return True
        
    @property
    def state_dim(self) -> int:
        """Dimension of the state space."""
        raise NotImplementedError("Subclasses should implement state_dim")
        
    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        raise NotImplementedError("Subclasses should implement action_dim")


class RewardFunction(ABC):
    """
    Abstract base class for reward/cost functions.
    
    Represents instantaneous rewards: r(x,u,t) and terminal rewards: Φ(x(T))
    """
    
    @abstractmethod
    def compute_reward(self, state: State, action: Action, 
                      time: Optional[float] = None) -> Reward:
        """
        Compute instantaneous reward.
        
        Args:
            state: Current state
            action: Applied action  
            time: Current time
            
        Returns:
            Reward value
        """
        pass
    
    def compute_terminal_reward(self, state: State) -> Reward:
        """
        Compute terminal reward (default: zero).
        
        Args:
            state: Terminal state
            
        Returns:
            Terminal reward
        """
        return Reward(value=jnp.array(0.0))
    
    def batch_compute_reward(self, states: State, actions: Action,
                           times: Optional[jnp.ndarray] = None) -> Reward:
        """
        Compute batch of rewards (default implementation).
        
        Args:
            states: Batch of states
            actions: Batch of actions
            times: Array of times
            
        Returns:
            Batch of rewards
        """
        import jax
        
        if times is not None:
            batch_fn = jax.vmap(self.compute_reward, in_axes=(0, 0, 0))
            return batch_fn(states, actions, times)
        else:
            batch_fn = jax.vmap(self.compute_reward, in_axes=(0, 0, None))
            return batch_fn(states, actions, time=None)


# Type aliases for common use cases
StateArray = jnp.ndarray  # For backward compatibility
ActionArray = jnp.ndarray  # For backward compatibility
TrajectoryData = Dict[str, jnp.ndarray]  # For storing trajectory information


def create_state(data: jnp.ndarray, time: Optional[float] = None, 
                **metadata) -> State:
    """
    Convenience function to create State objects.
    
    Args:
        data: State data array
        time: Current time
        **metadata: Additional metadata as keyword arguments
        
    Returns:
        State object
    """
    meta_dict = metadata if metadata else None
    return State(data=data, time=time, metadata=meta_dict)


def create_action(data: jnp.ndarray, **metadata) -> Action:
    """
    Convenience function to create Action objects.
    
    Args:
        data: Action data array
        **metadata: Additional metadata as keyword arguments
        
    Returns:
        Action object
    """
    meta_dict = metadata if metadata else None
    return Action(data=data, metadata=meta_dict)
"""
Control System Factory

This module provides a unified factory interface for creating control systems
using the control theory framework. It simplifies the creation of different
control configurations and provides preset configurations for common use cases.

Key Components:
    - ControlSystemFactory: Main factory class for creating control systems
    - ControlSystemConfig: Configuration class for system parameters
    - Preset configurations for common scenarios (Neumann-Voss, CE, etc.)

Usage:
    factory = ControlSystemFactory(config)
    system = factory.create_system('neumann_voss_2022')
    system = factory.create_custom_system(policy_type='riccati', environment_type='optimal_execution')
"""

from typing import Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import jax.numpy as jnp
from jax import random

from .core import ControlPolicy, State, Action
from .policies import RiccatiPolicy, NeuralPolicy, NetworkArchitectureRegistry
from .environments import ControlEnvironment, OptimalExecutionEnvironment
from .value_functions import ValueFunctionSolver, GeneralRiccatiSolver
from .trajectory import TrajectoryGenerator
from ..utils import Config, get_config, RiccatiSolver


@dataclass
class ControlSystemConfig:
    """
    Configuration class for control systems.
    
    Provides a structured way to specify system parameters and components.
    """
    # System identification
    name: str
    description: str
    
    # Core components
    policy_type: str  # 'riccati', 'neural', 'custom'
    environment_type: str  # 'optimal_execution', 'custom'
    value_solver_type: str  # 'riccati', 'hjb', 'neural'
    
    # Policy parameters
    policy_params: Dict[str, Any]
    
    # Environment parameters
    environment_params: Dict[str, Any]
    
    # Value solver parameters
    value_solver_params: Dict[str, Any]
    
    # System configuration
    use_trajectory_generator: bool = True
    compile_trajectory_gen: bool = True
    
    # Performance settings
    batch_size: int = 64
    default_n_steps: int = 200


class ControlSystemFactory:
    """
    Factory for creating control systems with the control theory framework.
    
    Provides both preset configurations and custom system creation capabilities.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize control system factory.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else get_config()
        self.riccati_solver = RiccatiSolver(self.config)
        
        # Registry of preset configurations
        self._presets = {}
        self._register_presets()
    
    def _register_presets(self):
        """Register preset control system configurations."""
        
        # Neumann-Voß 2022 configuration
        self._presets['neumann_voss_2022'] = ControlSystemConfig(
            name="Neumann-Voß 2022",
            description="Optimal execution using mean regime parameters",
            policy_type="riccati",
            environment_type="optimal_execution",
            value_solver_type="riccati",
            policy_params={
                "lambda_func": 0.5 * (self.config.LAMBDA_L + self.config.LAMBDA_H),  # Mean lambda
                "rho": self.config.RHO,
                "state_indices": {"X": 2}
            },
            environment_params={},
            value_solver_params={}
        )
        
        # Certainty Equivalent configuration
        def ce_lambda_func(state: State, time: Optional[float]) -> float:
            """Belief-dependent lambda for CE control."""
            p = state.data[3] if state.data.ndim == 1 else state.data[:, 3]
            return p * self.config.LAMBDA_L + (1.0 - p) * self.config.LAMBDA_H
        
        self._presets['certainty_equivalent'] = ControlSystemConfig(
            name="Certainty Equivalent",
            description="Belief-dependent control using expected regime parameters",
            policy_type="riccati",
            environment_type="optimal_execution",
            value_solver_type="riccati",
            policy_params={
                "lambda_func": ce_lambda_func,
                "rho": self.config.RHO,
                "state_indices": {"X": 2}
            },
            environment_params={},
            value_solver_params={}
        )
        
        # Neural policy configuration (template)
        self._presets['neural_gaussian'] = ControlSystemConfig(
            name="Neural Gaussian Policy",
            description="Neural network policy with Gaussian action distribution",
            policy_type="neural",
            environment_type="optimal_execution",
            value_solver_type="neural",
            policy_params={
                "network_architecture": "gaussian_policy",
                "policy_type": "gaussian",
                "action_bounds": (-5.0, 5.0)
            },
            environment_params={},
            value_solver_params={
                "network_architecture": "feedforward",
                "hidden_dims": (64, 64)
            }
        )
    
    def list_presets(self) -> Dict[str, str]:
        """
        List available preset configurations.
        
        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {name: config.description for name, config in self._presets.items()}
    
    def get_preset_config(self, preset_name: str) -> ControlSystemConfig:
        """
        Get preset configuration by name.
        
        Args:
            preset_name: Name of the preset configuration
            
        Returns:
            Preset configuration
            
        Raises:
            ValueError: If preset not found
        """
        if preset_name not in self._presets:
            available = list(self._presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        return self._presets[preset_name]
    
    def create_system(self, preset_name: str, **overrides) -> 'ControlSystem':
        """
        Create control system from preset configuration.
        
        Args:
            preset_name: Name of the preset configuration
            **overrides: Parameter overrides for customization
            
        Returns:
            Configured control system
        """
        config = self.get_preset_config(preset_name)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif key in config.policy_params:
                config.policy_params[key] = value
            elif key in config.environment_params:
                config.environment_params[key] = value
            elif key in config.value_solver_params:
                config.value_solver_params[key] = value
        
        return self._create_system_from_config(config)
    
    def create_custom_system(self, 
                           policy_type: str,
                           environment_type: str = "optimal_execution",
                           value_solver_type: str = "riccati",
                           **params) -> 'ControlSystem':
        """
        Create custom control system with specified components.
        
        Args:
            policy_type: Type of policy to create
            environment_type: Type of environment to create
            value_solver_type: Type of value solver to create
            **params: Additional parameters for components
            
        Returns:
            Custom control system
        """
        config = ControlSystemConfig(
            name="Custom System",
            description=f"Custom {policy_type} policy with {environment_type} environment",
            policy_type=policy_type,
            environment_type=environment_type,
            value_solver_type=value_solver_type,
            policy_params=params.get('policy_params', {}),
            environment_params=params.get('environment_params', {}),
            value_solver_params=params.get('value_solver_params', {}),
            **{k: v for k, v in params.items() if k not in ['policy_params', 'environment_params', 'value_solver_params']}
        )
        
        return self._create_system_from_config(config)
    
    def _create_system_from_config(self, config: ControlSystemConfig) -> 'ControlSystem':
        """
        Create control system from configuration.
        
        Args:
            config: System configuration
            
        Returns:
            Configured control system
        """
        # Create components
        policy = self._create_policy(config)
        environment = self._create_environment(config)
        value_solver = self._create_value_solver(config)
        
        # Create trajectory generator if requested
        trajectory_generator = None
        if config.use_trajectory_generator:
            trajectory_generator = TrajectoryGenerator(
                policy=policy,
                environment=environment,
                compile_trajectory_gen=config.compile_trajectory_gen
            )
        
        return ControlSystem(
            config=config,
            policy=policy,
            environment=environment,
            value_solver=value_solver,
            trajectory_generator=trajectory_generator,
            system_config=self.config
        )
    
    def _create_policy(self, config: ControlSystemConfig) -> ControlPolicy:
        """Create policy from configuration."""
        if config.policy_type == "riccati":
            return RiccatiPolicy(
                riccati_solver=self.riccati_solver,
                **config.policy_params
            )
        elif config.policy_type == "neural":
            # Create neural network
            network_arch = config.policy_params.get('network_architecture', 'gaussian_policy')
            network = NetworkArchitectureRegistry.create_network(network_arch)
            
            # Initialize network parameters (would need training in practice)
            key = random.PRNGKey(42)
            dummy_input = jnp.ones((self.config.STATE_DIM,))
            params = network.init(key, dummy_input)
            
            return NeuralPolicy(
                network=network,
                params=params,
                policy_type=config.policy_params.get('policy_type', 'gaussian'),
                action_bounds=config.policy_params.get('action_bounds')
            )
        else:
            raise ValueError(f"Unknown policy type: {config.policy_type}")
    
    def _create_environment(self, config: ControlSystemConfig) -> ControlEnvironment:
        """Create environment from configuration."""
        if config.environment_type == "optimal_execution":
            return OptimalExecutionEnvironment(self.config)
        else:
            raise ValueError(f"Unknown environment type: {config.environment_type}")
    
    def _create_value_solver(self, config: ControlSystemConfig) -> ValueFunctionSolver:
        """Create value solver from configuration."""
        if config.value_solver_type == "riccati":
            return GeneralRiccatiSolver(self.config)
        else:
            raise ValueError(f"Unknown value solver type: {config.value_solver_type}")


class ControlSystem:
    """
    Complete control system with all components.
    
    Provides a unified interface for interacting with the control system,
    including policy evaluation, trajectory generation, and performance analysis.
    """
    
    def __init__(self,
                 config: ControlSystemConfig,
                 policy: ControlPolicy,
                 environment: ControlEnvironment,
                 value_solver: ValueFunctionSolver,
                 trajectory_generator: Optional[TrajectoryGenerator] = None,
                 system_config: Optional[Config] = None):
        """
        Initialize control system.
        
        Args:
            config: System configuration
            policy: Control policy
            environment: Environment
            value_solver: Value function solver
            trajectory_generator: Optional trajectory generator
            system_config: System configuration
        """
        self.config = config
        self.policy = policy
        self.environment = environment
        self.value_solver = value_solver
        self.trajectory_generator = trajectory_generator
        self.system_config = system_config or get_config()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and configuration."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "policy_type": self.config.policy_type,
            "environment_type": self.config.environment_type,
            "value_solver_type": self.config.value_solver_type,
            "policy_stochastic": self.policy.is_stochastic,
            "policy_time_varying": self.policy.is_time_varying,
            "has_trajectory_generator": self.trajectory_generator is not None,
            "batch_size": self.config.batch_size,
            "default_n_steps": self.config.default_n_steps
        }
    
    def evaluate_policy(self, 
                       key: random.PRNGKey,
                       num_trajectories: int = None,
                       n_steps: int = None) -> Dict[str, Any]:
        """
        Evaluate policy performance.
        
        Args:
            key: Random key
            num_trajectories: Number of trajectories (uses config default if None)
            n_steps: Number of steps per trajectory (uses config default if None)
            
        Returns:
            Performance evaluation results
        """
        if self.trajectory_generator is None:
            raise ValueError("No trajectory generator available for evaluation")
        
        num_trajectories = num_trajectories or self.config.batch_size
        n_steps = n_steps or self.config.default_n_steps
        
        # Generate batch trajectories
        batch_trajectories = self.trajectory_generator.generate_batch_trajectories(
            batch_size=num_trajectories,
            n_steps=n_steps,
            key=key
        )
        
        # Compute performance metrics
        from .trajectory import TrajectoryAnalyzer
        analysis = TrajectoryAnalyzer.analyze_batch_trajectories(batch_trajectories)
        
        # Add system information
        analysis.update({
            "system_name": self.config.name,
            "policy_type": self.config.policy_type,
            "num_trajectories": num_trajectories,
            "n_steps": n_steps
        })
        
        return analysis
    
    def compute_action(self, state: State, time: Optional[float] = None, 
                      key: Optional[random.PRNGKey] = None) -> Action:
        """
        Compute action for given state.
        
        Args:
            state: Current state
            time: Current time
            key: Random key for stochastic policies
            
        Returns:
            Control action
        """
        return self.policy.compute_action(state, time, key)
    
    def step_environment(self, key: random.PRNGKey, state: State, action: Action) -> tuple:
        """
        Step environment with given state and action.
        
        Args:
            key: Random key
            state: Current state
            action: Control action
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        return self.environment.step(key, state, action)


# Factory instance for easy access
default_factory = None

def get_factory(config: Optional[Config] = None) -> ControlSystemFactory:
    """Get default factory instance."""
    global default_factory
    if default_factory is None or config is not None:
        default_factory = ControlSystemFactory(config)
    return default_factory


# Convenience functions
def create_neumann_voss_system(config: Optional[Config] = None) -> ControlSystem:
    """Create Neumann-Voß 2022 control system."""
    factory = get_factory(config)
    return factory.create_system('neumann_voss_2022')


def create_ce_system(config: Optional[Config] = None) -> ControlSystem:
    """Create Certainty Equivalent control system."""
    factory = get_factory(config)
    return factory.create_system('certainty_equivalent')


def create_neural_system(config: Optional[Config] = None) -> ControlSystem:
    """Create neural policy control system."""
    factory = get_factory(config)
    return factory.create_system('neural_gaussian')
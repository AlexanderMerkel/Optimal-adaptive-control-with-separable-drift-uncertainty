"""
REINFORCE Agent for Optimal Execution with Regime Uncertainty

Implements the REINFORCE algorithm using the corrected mathematical foundation
with state variables [Y, X, p, α_l, α_h] and proper accumulator dynamics.

This implementation trains for 10,000 episodes as specified in the paper requirements
and integrates with the control_theory framework.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import random
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np

from .config import OptimalExecutionConfig, default_config
from .environment import OptimalExecutionEnv
from .policies import Policy


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training."""
    
    # Training parameters
    n_episodes: int = 10000
    learning_rate: float = 1e-3
    batch_size: int = 64
    
    # Network architecture
    hidden_dim: int = 128
    n_layers: int = 3
    
    # Policy parameters
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_scale: float = 10.0  # Scale for actions
    
    # Training monitoring
    log_interval: int = 500
    eval_interval: int = 1000
    eval_episodes: int = 100


class PolicyNetwork(nn.Module):
    """Neural network for REINFORCE policy."""
    
    config: REINFORCEConfig
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass through policy network.
        
        Args:
            x: Input state [Y, X, p, α_l, α_h, t] - shape (6,)
            
        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        # Normalize inputs for stability
        x = x / jnp.array([100.0, 10.0, 1.0, 5.0, 5.0, 1.0])  # Rough normalization
        
        # Deep feedforward network
        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.hidden_dim)(x)
            x = nn.tanh(x)
        
        # Output mean and log_std for Gaussian policy
        mean = nn.Dense(1)(x).squeeze()
        log_std = nn.Dense(1)(x).squeeze()
        log_std = jnp.clip(log_std, self.config.log_std_min, self.config.log_std_max)
        
        return mean, log_std


class REINFORCEPolicy(Policy):
    """REINFORCE-trained policy for optimal execution."""
    
    def __init__(self, 
                 network: PolicyNetwork,
                 params: Dict[str, Any],
                 config: REINFORCEConfig,
                 problem_config: OptimalExecutionConfig = default_config):
        """Initialize REINFORCE policy."""
        self.network = network
        self.params = params
        self.config = config
        self.problem_config = problem_config
        
        # JIT compile for performance (network.apply is used directly)
    
    def __call__(self, state: jnp.ndarray, time: float) -> float:
        """Compute action using trained policy."""
        return self.get_action(state, time, deterministic=True)
    
    def get_action(self, state: jnp.ndarray, time: float, 
                   key: random.PRNGKey = None, deterministic: bool = False) -> float:
        """Get action from policy with optional stochasticity."""
        # Add time to state
        network_input = jnp.concatenate([state, jnp.array([time])])
        
        # Get policy parameters
        mean, log_std = self.network.apply({"params": self.params}, network_input)
        
        if deterministic or key is None:
            # Use mean for deterministic evaluation
            action = mean
        else:
            # Sample from policy
            std = jnp.exp(log_std)
            action = mean + std * random.normal(key)
        
        # Scale action
        action = action * self.config.action_scale
        
        # Apply constraints
        Y, X, p, alpha_l, alpha_h = state
        action = jnp.maximum(action, 0.0)  # Non-negative
        action = jnp.minimum(action, X / self.problem_config.dt)  # Inventory constraint
        
        return action
    
    def log_prob(self, state: jnp.ndarray, time: float, action: float) -> float:
        """Compute log probability of action."""
        network_input = jnp.concatenate([state, jnp.array([time])])
        mean, log_std = self.network.apply({"params": self.params}, network_input)
        
        # Unscale action for probability computation
        scaled_action = action / self.config.action_scale
        
        # Gaussian log probability
        std = jnp.exp(log_std)
        log_prob = -0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * ((scaled_action - mean) / std)**2
        
        return log_prob
    
    @property
    def name(self) -> str:
        return "REINFORCE"


class REINFORCEAgent:
    """REINFORCE agent for optimal execution training."""
    
    def __init__(self, 
                 problem_config: OptimalExecutionConfig = default_config,
                 reinforce_config: REINFORCEConfig = REINFORCEConfig()):
        """Initialize REINFORCE agent."""
        self.problem_config = problem_config
        self.config = reinforce_config
        
        # Environment
        self.env = OptimalExecutionEnv(problem_config)
        
        # Network
        self.network = PolicyNetwork(reinforce_config)
        
        # Initialize parameters
        key = random.PRNGKey(42)
        dummy_input = jnp.ones(6)  # [Y, X, p, α_l, α_h, t]
        init_params = self.network.init(key, dummy_input)
        self.params = init_params["params"]  # Extract just the params
        
        # Optimizer
        self.optimizer = optax.adam(reinforce_config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        # JIT compile training functions (rollout removed due to control flow)  
        self._update_policy = jax.jit(self._update_policy_impl)
    
    def train(self, key: random.PRNGKey) -> Dict[str, List[float]]:
        """
        Train REINFORCE policy for specified number of episodes.
        
        Args:
            key: Random key for training
            
        Returns:
            Training history with rewards and losses
        """
        history = {"rewards": [], "losses": [], "eval_rewards": []}
        
        print(f"Starting REINFORCE training for {self.config.n_episodes} episodes...")
        
        for episode in range(self.config.n_episodes):
            # Generate episode
            key, episode_key = random.split(key)
            episode_data = self._rollout_episode_impl(episode_key)
            
            # Update policy
            self.params, self.opt_state, loss = self._update_policy(
                self.params, self.opt_state, episode_data
            )
            
            # Record metrics
            total_reward = float(jnp.sum(episode_data["rewards"]))
            history["rewards"].append(total_reward)
            history["losses"].append(float(loss))
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(history["rewards"][-self.config.log_interval:])
                print(f"Episode {episode + 1:5d}: Avg Reward = {avg_reward:8.3f}, Loss = {loss:.6f}")
            
            # Evaluation
            if (episode + 1) % self.config.eval_interval == 0:
                key, eval_key = random.split(key)
                eval_reward = self._evaluate_policy(eval_key)
                history["eval_rewards"].append(float(eval_reward))
                print(f"Evaluation at episode {episode + 1}: Reward = {eval_reward:.3f}")
        
        print("REINFORCE training completed!")
        return history
    
    def _rollout_episode_impl(self, key: random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Generate a single episode rollout."""
        # Reset environment
        key, reset_key = random.split(key)
        state = self.env.reset(reset_key)
        
        # Storage
        n_steps = self.problem_config.n_steps
        states = jnp.zeros((n_steps, 5))
        actions = jnp.zeros(n_steps)
        rewards = jnp.zeros(n_steps)
        log_probs = jnp.zeros(n_steps)
        
        # Episode rollout
        for t in range(n_steps):
            current_time = t * self.problem_config.dt
            
            # Get action from policy
            key, action_key = random.split(key)
            network_input = jnp.concatenate([state, jnp.array([current_time])])
            mean, log_std = self.network.apply({"params": self.params}, network_input)
            
            # Sample action
            std = jnp.exp(log_std)
            action_raw = mean + std * random.normal(action_key)
            action = action_raw * self.config.action_scale
            
            # Apply constraints
            Y, X, p, alpha_l, alpha_h = state
            action = jnp.maximum(action, 0.0)
            action = jnp.minimum(action, X / self.problem_config.dt)
            
            # Compute log probability
            log_prob = -0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * ((action_raw - mean) / std)**2
            
            # Store
            states = states.at[t].set(state)
            actions = actions.at[t].set(action)
            log_probs = log_probs.at[t].set(log_prob)
            
            # Take environment step
            key, step_key = random.split(key)
            result = self.env.step(state, action, step_key)
            
            state = result.next_state
            rewards = rewards.at[t].set(result.reward)
            
            # Note: removed early termination for JAX compatibility
        
        # Add terminal reward
        terminal_reward = self.env.compute_terminal_reward(state)
        rewards = rewards.at[-1].add(terminal_reward)
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs
        }
    
    def _update_policy_impl(self, params, opt_state, episode_data):
        """Update policy parameters using REINFORCE."""
        def loss_fn(params):
            # REINFORCE loss: -E[∑ log π(a|s) * G]
            # where G is the discounted return
            returns = jnp.cumsum(episode_data["rewards"][::-1])[::-1]  # Undiscounted returns
            
            # Policy gradient loss
            advantages = returns - jnp.mean(returns)  # Baseline subtraction
            loss = -jnp.mean(episode_data["log_probs"] * advantages)
            
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss
    
    def _evaluate_policy(self, key: random.PRNGKey) -> float:
        """Evaluate current policy deterministically."""
        total_rewards = []
        
        for _ in range(self.config.eval_episodes):
            key, episode_key = random.split(key)
            state = self.env.reset(episode_key)
            
            episode_reward = 0.0
            for t in range(self.problem_config.n_steps):
                current_time = t * self.problem_config.dt
                
                # Deterministic action
                network_input = jnp.concatenate([state, jnp.array([current_time])])
                mean, _ = self.network.apply({"params": self.params}, network_input)
                action = mean * self.config.action_scale
                
                # Apply constraints
                Y, X, p, alpha_l, alpha_h = state
                action = jnp.maximum(action, 0.0)
                action = jnp.minimum(action, X / self.problem_config.dt)
                
                # Step
                key, step_key = random.split(key)
                result = self.env.step(state, action, step_key)
                
                state = result.next_state
                episode_reward += result.reward
                
                # Note: removed early termination for JAX compatibility
            
            # Terminal reward
            episode_reward += self.env.compute_terminal_reward(state)
            total_rewards.append(episode_reward)
        
        return jnp.mean(jnp.array(total_rewards))
    
    def create_policy(self) -> REINFORCEPolicy:
        """Create trained policy for comparison."""
        return REINFORCEPolicy(self.network, self.params, self.config, self.problem_config)


def train_reinforce_policy(problem_config: OptimalExecutionConfig = default_config,
                         reinforce_config: REINFORCEConfig = REINFORCEConfig(),
                         key: random.PRNGKey = random.PRNGKey(42)) -> REINFORCEPolicy:
    """
    Train REINFORCE policy for optimal execution.
    
    Args:
        problem_config: Problem configuration
        reinforce_config: REINFORCE training configuration
        key: Random key for training
        
    Returns:
        Trained REINFORCE policy
    """
    agent = REINFORCEAgent(problem_config, reinforce_config)
    history = agent.train(key)
    
    # Print final statistics
    final_rewards = history["rewards"][-100:]  # Last 100 episodes
    print("\nFinal Performance (last 100 episodes):")
    print(f"  Mean Reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    print(f"  Best Reward: {np.max(history['rewards']):.3f}")
    
    return agent.create_policy()
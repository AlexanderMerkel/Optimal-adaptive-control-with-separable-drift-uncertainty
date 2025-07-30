"""
REINFORCE Controller for Optimal Execution with Regime Uncertainty

This implements the REINFORCE algorithm with rigorous Wonham filtering for
optimal execution under regime uncertainty. The agent learns optimal policies
through policy gradient methods while simultaneously detecting regime changes
through price observations.

Mathematical Foundation:
- State Space: [t, S, X, p, A_l, A_h] (6D observable)
- Hidden regime drives true price dynamics
- Wonham filter updates belief p(t) based on innovations process
- REINFORCE learns policy π(a|s) through gradient ascent on expected return
"""

# Third-party imports
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from jax import random

# Local imports
from ...utils import get_config, WonhamFilter, StateManager, PriceDynamics, update_full_state


class REINFORCEController:
    """REINFORCE controller for optimal execution under regime uncertainty."""

    def __init__(self, config=None, hidden_dim=64, batch_size=64):
        """Initialize REINFORCE controller with centralized configuration."""
        self.config = config if config is not None else get_config()

        self.wonham_filter = WonhamFilter(self.config)
        self.state_manager = StateManager(self.config)
        self.price_dynamics = PriceDynamics(self.config)

        # Expose frequently used parameters for backward compatibility
        self.T = self.config.T
        self.N = self.config.N
        self.dt = self.config.dt

        # Network parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.policy = self.PolicyNetwork(hidden_dim)

        key = random.PRNGKey(42)
        dummy_state = jnp.zeros((1, 6))  # 6D observable state
        self.policy_params = self.policy.init(key, dummy_state)["params"]

        self.optimizer = optax.adam(1e-3)
        self.opt_state = self.optimizer.init(self.policy_params)

    class PolicyNetwork(nn.Module):
        """Neural network policy for optimal execution."""
        hidden_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
            mean = nn.Dense(1)(x)
            log_std = nn.Dense(1)(x)
            log_std = jnp.clip(log_std, -5.0, 2.0)
            return mean, log_std

    def reset_env_with_true_regime(self, key, batch_size=64):
        """Initialize environment with hidden true regime per trajectory."""
        return self.state_manager.initialize_batch(batch_size, key)

    def step_env_with_innovations(self, key, internal_state, actions, batch_size=64):
        """Environment step implementing rigorous Wonham filtering with innovations process."""
        key, subkey = random.split(key)
        dW = jnp.sqrt(self.config.dt) * random.normal(subkey, (batch_size,), dtype=jnp.float32)
        actions = actions.ravel()[:batch_size]

        next_internal, next_observable, rewards, info = update_full_state(internal_state, actions, dW, self.config)
        return next_internal, next_observable, rewards, jnp.zeros(batch_size, dtype=bool), info

    def rollout_with_filtering(self, key, policy_params, policy_apply, initial_internal, n_steps=200, batch_size=64):
        """Complete rollout with filtering dynamics and policy execution."""
        def scan_body(carry, _):
            key, internal_state = carry
            key, policy_key, env_key = random.split(key, 3)

            observable = internal_state[:, :6]  # Agent observes only 6D state (no true regime)
            mean, log_std = policy_apply({"params": policy_params}, observable)
            mean, log_std = mean[..., 0], log_std[..., 0]

            # Sample action using native JAX (no distrax dependency)
            std_normal = random.normal(policy_key, mean.shape)
            actions_raw = mean + jnp.exp(log_std) * std_normal
            actions = jnp.tanh(actions_raw) * 5.0  # Apply tanh squashing

            # Compute log probability manually (normal PDF + tanh correction)
            log_probs = -0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * ((actions_raw - mean) / jnp.exp(log_std))**2
            log_probs = log_probs - jnp.log(5.0 * (1 - jnp.tanh(actions_raw)**2) + 1e-6)

            next_internal, next_obs, reward, done, info = self.step_env_with_innovations(
                env_key, internal_state, actions, batch_size
            )

            return (key, next_internal), (observable, actions, reward, log_probs, info)

        (_, final_internal), (states, actions, rewards, log_probs, infos) = jax.lax.scan(
            scan_body, (key, initial_internal), None, length=n_steps
        )

        # Terminal reward
        final_obs = final_internal[:, :6]
        S_final, X_final = final_obs[:, 1], final_obs[:, 2]
        terminal_reward = S_final * X_final - self.config.C_TERMINAL * X_final**2
        rewards = rewards.at[-1, :].add(terminal_reward)

        return states, actions, rewards, log_probs, final_internal, infos

    def train_step(self, key, policy_params, opt_state, policy_apply, optimizer_update, n_steps=200, batch_size=64):
        """REINFORCE training step with policy gradient update."""
        key, rollout_key, reset_key = random.split(key, 3)
        initial_internal, _ = self.reset_env_with_true_regime(reset_key, batch_size)

        def loss_fn(params):
            _, _, rewards, log_probs, _, _ = self.rollout_with_filtering(
                rollout_key, params, policy_apply, initial_internal, n_steps, batch_size
            )

            total_profit = jnp.sum(rewards, axis=0)
            sum_log_probs = jnp.sum(log_probs, axis=0)
            loss = -jnp.mean(sum_log_probs * jax.lax.stop_gradient(total_profit))
            return loss, total_profit

        (loss, profits), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_params)
        updates, new_opt_state = optimizer_update(grads, opt_state, policy_params)
        new_params = optax.apply_updates(policy_params, updates)

        return new_params, new_opt_state, {'loss': loss, 'profit': jnp.mean(profits)}

    def train_policy(self, num_episodes=800, learning_rate=1e-3, seed=42, verbose=True):
        """Train the REINFORCE policy."""
        key = random.PRNGKey(seed)

        # Reset optimizer with specified learning rate
        if learning_rate != 1e-3:
            self.optimizer = optax.adam(learning_rate)
            self.opt_state = self.optimizer.init(self.policy_params)

        profits = []

        if verbose:
            print(f"Training REINFORCE policy for {num_episodes} episodes...")

        for episode in range(num_episodes):
            key, step_key = random.split(key)
            self.policy_params, self.opt_state, metrics = self.train_step(
                step_key, self.policy_params, self.opt_state,
                self.policy.apply, self.optimizer.update,
                self.N, self.batch_size
            )

            profits.append(float(metrics['profit']))

            if verbose and episode % 100 == 0:
                print(f"Episode {episode}: Profit={metrics['profit']:.2f}")

        if verbose:
            print(f"Training complete! Final profit: {profits[-1]:.2f}")

        return profits

    def evaluate_performance(self, key, num_trajectories=100, n_steps=200):
        """Evaluate REINFORCE policy performance."""
        key, reset_key, rollout_key = random.split(key, 3)

        initial_internal, _ = self.reset_env_with_true_regime(reset_key, num_trajectories)

        # Run REINFORCE policy
        states, actions, rewards, log_probs, final_internal, infos = self.rollout_with_filtering(
            rollout_key, self.policy_params, self.policy.apply, initial_internal, n_steps, num_trajectories
        )

        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        true_regimes_np = np.array(final_internal[:, 6])

        # Performance metrics
        total_profits = np.sum(rewards_np, axis=0)
        beliefs = states_np[:, :, 3]  # p(t)
        final_beliefs = beliefs[-1, :]

        # Regime detection accuracy
        predicted_regimes = (final_beliefs < 0.5).astype(int)
        true_regimes_int = true_regimes_np.astype(int)
        accuracy = np.mean(predicted_regimes == true_regimes_int)

        results = {
            'method': 'REINFORCE',
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

    def get_trained_policy(self):
        """Get the trained policy parameters and apply function."""
        return self.policy_params, self.policy.apply


if __name__ == "__main__":
    # Test the REINFORCE controller
    print("Testing REINFORCE Controller...")

    controller = REINFORCEController(hidden_dim=64, batch_size=64)

    # Train policy
    profits = controller.train_policy(num_episodes=500, verbose=True)

    # Evaluate performance
    key = random.PRNGKey(123)
    results = controller.evaluate_performance(key, num_trajectories=50)

    print("\nREINFORCE Results:")
    print(f"  Mean profit: {results['mean_profit']:.4f} ± {results['std_profit']:.4f}")
    print(f"  Regime detection accuracy: {results['regime_accuracy']:.1%}")

"""
Policy Comparison and Evaluation Utilities for Paper Methods

Comprehensive comparison framework implementing the exact controls from
"Optimal adaptive control with separable drift uncertainty":
1. Deep Reinforcement Learning (REINFORCE with 10,000 episodes)
2. Certainty Equivalent control (explicit solution)
3. Oracle control (upper bound with perfect information)

Includes convergence analysis, learning curves, and statistical comparisons.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

from .environment import OptimalExecutionEnv
from .policies import Policy, CertaintyEquivalentPolicy, NaivePolicy, OraclePolicy
from .config import OptimalExecutionConfig, default_config
from .reinforce_agent import train_reinforce_policy, REINFORCEConfig


@dataclass
class PolicyResult:
    """Results from evaluating a single policy."""
    name: str
    total_rewards: jnp.ndarray     # Total rewards across episodes
    final_inventories: jnp.ndarray # Final inventory levels
    trajectories: List[Dict]       # Full trajectory data
    metrics: Dict[str, float]      # Computed performance metrics


class PaperMethodsComparator:
    """
    Comprehensive comparator implementing all methods from the paper.
    
    Compares:
    1. REINFORCE agent (10,000 episodes training with learning curves)
    2. Certainty Equivalent control (explicit solution)
    3. Oracle control (perfect information upper bound)
    
    Provides convergence analysis and statistical comparisons.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config,
                 reinforce_config: REINFORCEConfig = None):
        """Initialize paper methods comparator."""
        self.config = config
        self.env = OptimalExecutionEnv(config)
        
        # Training configuration
        self.reinforce_config = reinforce_config or REINFORCEConfig(n_episodes=10000)
    
    def compare_all_methods(self, 
                          key: random.PRNGKey = random.PRNGKey(42),
                          n_evaluation_episodes: int = 500,
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Train and compare all methods from the paper.
        
        Args:
            key: Random key for reproducible results
            n_evaluation_episodes: Episodes for final evaluation
            verbose: Whether to print training progress
            
        Returns:
            Comprehensive comparison results including learning curves
        """
        results = {}
        keys = random.split(key, 3)
        
        if verbose:
            print("="*60)
            print("OPTIMAL CONTROL COMPARISON")
            print("Methods: REINFORCE, Certainty Equivalent, Oracle")
            print("="*60)
        
        # 1. Train REINFORCE Agent with Learning Curves
        if verbose:
            print("\n1. Training REINFORCE Agent (10,000 episodes)...")
        reinforce_policy, learning_history = self._train_reinforce_with_curves(keys[0], verbose)
        results['reinforce'] = self._evaluate_policy_detailed(
            reinforce_policy, keys[0], n_evaluation_episodes
        )
        results['reinforce']['method'] = 'REINFORCE'
        results['reinforce']['learning_history'] = learning_history
        
        # 2. Certainty Equivalent Control
        if verbose:
            print("\n2. Evaluating Certainty Equivalent Control...")
        ce_policy = CertaintyEquivalentPolicy(self.config)
        results['certainty_equivalent'] = self._evaluate_policy_detailed(
            ce_policy, keys[1], n_evaluation_episodes
        )
        results['certainty_equivalent']['method'] = 'Certainty Equivalent'
        
        # 3. Oracle Control (Upper Bound)
        if verbose:
            print("\n3. Evaluating Oracle Control...")
        oracle_policy = OraclePolicy(self.config)
        results['oracle'] = self._evaluate_oracle_detailed(
            oracle_policy, keys[2], n_evaluation_episodes
        )
        results['oracle']['method'] = 'Oracle'
        
        if verbose:
            print("\n" + "="*60)
            print("COMPARISON COMPLETED")
            self._print_summary(results)
        
        return results
    
    def _train_reinforce_with_curves(self, key: random.PRNGKey, verbose: bool = True):
        """Train REINFORCE and capture learning curves."""
        from .reinforce_agent import REINFORCEAgent
        
        agent = REINFORCEAgent(self.config, self.reinforce_config)
        history = agent.train(key)
        
        return agent.create_policy(), history
    
    def _evaluate_policy_detailed(self, policy: Policy, key: random.PRNGKey, 
                                n_episodes: int) -> Dict[str, Any]:
        """Detailed evaluation of a policy."""
        episode_keys = random.split(key, n_episodes)
        
        total_rewards = []
        final_inventories = []
        trajectories = []
        
        for i, episode_key in enumerate(episode_keys):
            # Generate trajectory
            trajectory = self.env.generate_trajectory(policy, key=episode_key)
            trajectories.append(trajectory)
            
            # Extract metrics
            total_rewards.append(float(trajectory['total_reward']))
            final_inventories.append(float(trajectory['final_state'][1]))  # X
            
            if (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{n_episodes} episodes")
        
        total_rewards = jnp.array(total_rewards)
        final_inventories = jnp.array(final_inventories)
        
        return {
            'total_rewards': total_rewards,
            'final_inventories': final_inventories,
            'trajectories': trajectories[:10],  # Store first 10 for plotting
            'metrics': self._compute_detailed_metrics(total_rewards, final_inventories)
        }
    
    def _evaluate_oracle_detailed(self, oracle_policy: OraclePolicy, key: random.PRNGKey,
                                n_episodes: int) -> Dict[str, Any]:
        """Evaluate oracle policy with true regime access."""
        episode_keys = random.split(key, n_episodes)
        
        total_rewards = []
        final_inventories = []
        trajectories = []
        
        for i, episode_key in enumerate(episode_keys):
            # Reset environment and set oracle regime
            env_key, traj_key = random.split(episode_key)
            self.env.reset(env_key)
            oracle_policy.set_true_regime(self.env.true_regime)
            
            # Generate trajectory
            trajectory = self.env.generate_trajectory(oracle_policy, key=traj_key)
            trajectories.append(trajectory)
            
            # Extract metrics
            total_rewards.append(float(trajectory['total_reward']))
            final_inventories.append(float(trajectory['final_state'][1]))
            
            if (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{n_episodes} episodes")
        
        total_rewards = jnp.array(total_rewards)
        final_inventories = jnp.array(final_inventories)
        
        return {
            'total_rewards': total_rewards,
            'final_inventories': final_inventories,
            'trajectories': trajectories[:10],
            'metrics': self._compute_detailed_metrics(total_rewards, final_inventories)
        }
    
    def _compute_detailed_metrics(self, total_rewards: jnp.ndarray, 
                                final_inventories: jnp.ndarray) -> Dict[str, float]:
        """Compute detailed performance metrics."""
        return {
            'mean_reward': float(jnp.mean(total_rewards)),
            'std_reward': float(jnp.std(total_rewards)),
            'min_reward': float(jnp.min(total_rewards)),
            'max_reward': float(jnp.max(total_rewards)),
            'median_reward': float(jnp.median(total_rewards)),
            'reward_q25': float(jnp.percentile(total_rewards, 25)),
            'reward_q75': float(jnp.percentile(total_rewards, 75)),
            'mean_final_inventory': float(jnp.mean(jnp.abs(final_inventories))),
            'std_final_inventory': float(jnp.std(final_inventories)),
            'liquidation_rate': float(jnp.mean(jnp.abs(final_inventories) < 0.1)),
            'sharpe_ratio': float(jnp.mean(total_rewards) / (jnp.std(total_rewards) + 1e-8)),
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comparison summary."""
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Method':<20} {'Mean Reward':<12} {'Std Reward':<12} {'Sharpe Ratio':<12}")
        print("-" * 80)
        
        for method_name, result in results.items():
            metrics = result['metrics']
            print(f"{result['method']:<20} "
                  f"{metrics['mean_reward']:<12.3f} "
                  f"{metrics['std_reward']:<12.3f} "
                  f"{metrics['sharpe_ratio']:<12.3f}")
        
        print("-" * 80)
        
        # Rank methods by mean reward
        ranking = sorted(results.items(), 
                        key=lambda x: x[1]['metrics']['mean_reward'], 
                        reverse=True)
        
        print("\nRANKING (by mean reward):")
        for i, (method_name, result) in enumerate(ranking):
            print(f"{i+1}. {result['method']}: {result['metrics']['mean_reward']:.3f}")
    
    def plot_comprehensive_results(self, results: Dict[str, Any], 
                                 save_path: Optional[str] = None, 
                                 show: bool = True):
        """Create comprehensive visualization of all results."""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. REINFORCE Learning Curve
        if 'reinforce' in results and 'learning_history' in results['reinforce']:
            ax1 = plt.subplot(2, 3, 1)
            history = results['reinforce']['learning_history']
            episodes = np.arange(len(history['rewards']))
            
            # Smooth learning curve
            window_size = min(100, len(history['rewards']) // 10)
            window_size = max(1, window_size)  # Ensure window_size is at least 1
            
            ax1.plot(episodes, history['rewards'], alpha=0.3, color='blue', label='Raw')
            
            if window_size > 1 and len(history['rewards']) > window_size:
                smoothed_rewards = np.convolve(history['rewards'], 
                                             np.ones(window_size)/window_size, 
                                             mode='valid')
                smoothed_episodes = episodes[window_size-1:]
                ax1.plot(smoothed_episodes, smoothed_rewards, color='blue', linewidth=2, label='Smoothed')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('REINFORCE Learning Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Reward Comparison
        ax2 = plt.subplot(2, 3, 2)
        methods = []
        means = []
        stds = []
        
        for method_name, result in results.items():
            methods.append(result['method'])
            means.append(result['metrics']['mean_reward'])
            stds.append(result['metrics']['std_reward'])
        
        colors = ['red', 'blue', 'green', 'orange'][:len(methods)]
        bars = ax2.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax2.set_ylabel('Mean Total Reward')
        ax2.set_title('Performance Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Reward Distributions
        ax3 = plt.subplot(2, 3, 3)
        for i, (method_name, result) in enumerate(results.items()):
            rewards = result['total_rewards']
            ax3.hist(rewards, bins=30, alpha=0.6, label=result['method'], 
                    color=colors[i % len(colors)])
        
        ax3.set_xlabel('Total Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sample Trajectories - Price
        ax4 = plt.subplot(2, 3, 4)
        time_grid = self.config.time_grid
        
        for i, (method_name, result) in enumerate(results.items()):
            if 'trajectories' in result and result['trajectories']:
                traj = result['trajectories'][0]  # First trajectory
                prices = traj['states'][:, 0]  # Y
                ax4.plot(time_grid[:len(prices)], prices, 
                        label=result['method'], color=colors[i % len(colors)])
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Price (Y)')
        ax4.set_title('Sample Price Trajectories')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sample Trajectories - Inventory
        ax5 = plt.subplot(2, 3, 5)
        
        for i, (method_name, result) in enumerate(results.items()):
            if 'trajectories' in result and result['trajectories']:
                traj = result['trajectories'][0]
                inventory = traj['states'][:, 1]  # X
                ax5.plot(time_grid[:len(inventory)], inventory,
                        label=result['method'], color=colors[i % len(colors)])
        
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Inventory (X)')
        ax5.set_title('Sample Inventory Trajectories')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistical Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "STATISTICAL SUMMARY\n" + "="*30 + "\n\n"
        for method_name, result in results.items():
            metrics = result['metrics']
            summary_text += f"{result['method']}:\n"
            summary_text += f"  Mean: {metrics['mean_reward']:.3f}\n"
            summary_text += f"  Std:  {metrics['std_reward']:.3f}\n"
            summary_text += f"  Sharpe: {metrics['sharpe_ratio']:.3f}\n\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()


class PolicyComparator:
    """
    Legacy comparator for backward compatibility.
    
    Ensures fair comparison by using same random seeds and environments
    for all policies.
    """
    
    def __init__(self, config: OptimalExecutionConfig = default_config):
        """Initialize comparator with configuration."""
        self.config = config
        self.env = OptimalExecutionEnv(config)
    
    def compare_policies(self, 
                        policies: List[Policy],
                        n_episodes: int = 100,
                        key: random.PRNGKey = random.PRNGKey(0),
                        verbose: bool = True) -> Dict[str, PolicyResult]:
        """
        Compare multiple policies over many episodes.
        
        Args:
            policies: List of policies to compare
            n_episodes: Number of episodes per policy
            key: Random key for reproducible results
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping policy names to results
        """
        results = {}
        
        # Generate fixed random seeds for fair comparison
        episode_keys = random.split(key, n_episodes)
        
        for policy in policies:
            if verbose:
                print(f"Evaluating {policy.name}...")
            
            # Special handling for Oracle policy
            if hasattr(policy, 'set_true_regime'):
                results[policy.name] = self._evaluate_oracle_policy(
                    policy, episode_keys, verbose
                )
            else:
                results[policy.name] = self._evaluate_policy(
                    policy, episode_keys, verbose
                )
        
        if verbose:
            print("Policy comparison completed!")
        
        return results
    
    def _evaluate_policy(self, policy: Policy, episode_keys: jnp.ndarray, 
                        verbose: bool = False) -> PolicyResult:
        """Evaluate a single policy over multiple episodes."""
        n_episodes = len(episode_keys)
        total_rewards = jnp.zeros(n_episodes)
        final_inventories = jnp.zeros(n_episodes)
        trajectories = []
        
        for i, key in enumerate(episode_keys):
            # Generate trajectory
            trajectory = self.env.generate_trajectory(policy, key=key)
            trajectories.append(trajectory)
            
            # Extract metrics
            total_rewards = total_rewards.at[i].set(trajectory['total_reward'])
            final_inventories = final_inventories.at[i].set(trajectory['final_state'][1])  # X
            
            if verbose and (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_episodes} episodes")
        
        # Compute performance metrics
        metrics = self._compute_metrics(total_rewards, final_inventories, trajectories)
        
        return PolicyResult(
            name=policy.name,
            total_rewards=total_rewards,
            final_inventories=final_inventories,
            trajectories=trajectories,
            metrics=metrics
        )
    
    def _evaluate_oracle_policy(self, policy: Policy, episode_keys: jnp.ndarray,
                               verbose: bool = False) -> PolicyResult:
        """Evaluate oracle policy (needs access to true regime)."""
        n_episodes = len(episode_keys)
        total_rewards = jnp.zeros(n_episodes)
        final_inventories = jnp.zeros(n_episodes)
        trajectories = []
        
        for i, key in enumerate(episode_keys):
            # Reset environment and get true regime
            env_key, traj_key = random.split(key)
            self.env.reset(env_key)
            
            # Set true regime for oracle
            policy.set_true_regime(self.env.true_regime)
            
            # Generate trajectory
            trajectory = self.env.generate_trajectory(policy, key=traj_key)
            trajectories.append(trajectory)
            
            # Extract metrics
            total_rewards = total_rewards.at[i].set(trajectory['total_reward'])
            final_inventories = final_inventories.at[i].set(trajectory['final_state'][1])
            
            if verbose and (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_episodes} episodes")
        
        # Compute metrics
        metrics = self._compute_metrics(total_rewards, final_inventories, trajectories)
        
        return PolicyResult(
            name=policy.name,
            total_rewards=total_rewards,
            final_inventories=final_inventories,
            trajectories=trajectories,
            metrics=metrics
        )
    
    def _compute_metrics(self, total_rewards: jnp.ndarray, 
                        final_inventories: jnp.ndarray,
                        trajectories: List[Dict]) -> Dict[str, float]:
        """Compute performance metrics for a policy."""
        metrics = {
            # Reward statistics
            'mean_reward': float(jnp.mean(total_rewards)),
            'std_reward': float(jnp.std(total_rewards)),
            'min_reward': float(jnp.min(total_rewards)),
            'max_reward': float(jnp.max(total_rewards)),
            'median_reward': float(jnp.median(total_rewards)),
            
            # Inventory statistics
            'mean_final_inventory': float(jnp.mean(jnp.abs(final_inventories))),
            'std_final_inventory': float(jnp.std(final_inventories)),
            'liquidation_rate': float(jnp.mean(jnp.abs(final_inventories) < 0.1)),
            
            # Risk metrics
            'sharpe_ratio': float(jnp.mean(total_rewards) / (jnp.std(total_rewards) + 1e-8)),
            'reward_q25': float(jnp.percentile(total_rewards, 25)),
            'reward_q75': float(jnp.percentile(total_rewards, 75)),
        }
        
        # Additional metrics from trajectory info
        if trajectories:
            execution_revenues = []
            inventory_costs = []
            
            for traj in trajectories:
                if 'infos' in traj:
                    exec_rev = sum([info.get('execution_revenue', 0) for info in traj['infos']])
                    inv_cost = sum([info.get('inventory_cost', 0) for info in traj['infos']])
                    execution_revenues.append(exec_rev)
                    inventory_costs.append(inv_cost)
            
            if execution_revenues:
                metrics['mean_execution_revenue'] = float(jnp.mean(jnp.array(execution_revenues)))
                metrics['mean_inventory_cost'] = float(jnp.mean(jnp.array(inventory_costs)))
        
        return metrics
    
    def statistical_comparison(self, results: Dict[str, PolicyResult]) -> Dict[str, Any]:
        """
        Perform statistical comparison between policies.
        
        Args:
            results: Results from compare_policies
            
        Returns:
            Statistical comparison results
        """
        policy_names = list(results.keys())
        n_policies = len(policy_names)
        
        # Pairwise comparisons
        comparisons = {}
        for i in range(n_policies):
            for j in range(i + 1, n_policies):
                name_i, name_j = policy_names[i], policy_names[j]
                
                rewards_i = results[name_i].total_rewards
                rewards_j = results[name_j].total_rewards
                
                # Simple statistical test (difference in means)
                diff_mean = float(jnp.mean(rewards_i - rewards_j))
                diff_std = float(jnp.std(rewards_i - rewards_j))
                t_stat = diff_mean / (diff_std / jnp.sqrt(len(rewards_i)) + 1e-8)
                
                comparisons[f"{name_i}_vs_{name_j}"] = {
                    'mean_difference': diff_mean,
                    'std_difference': diff_std,
                    't_statistic': float(t_stat),
                    'significant': abs(t_stat) > 1.96  # Rough 95% confidence
                }
        
        # Overall ranking
        mean_rewards = {name: results[name].metrics['mean_reward'] 
                       for name in policy_names}
        ranking = sorted(mean_rewards.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'pairwise_comparisons': comparisons,
            'ranking': ranking,
            'mean_rewards': mean_rewards
        }
    
    def plot_results(self, results: Dict[str, PolicyResult], 
                    save_path: str = None, show: bool = True):
        """
        Create visualization plots for policy comparison.
        
        Args:
            results: Results from compare_policies
            save_path: Path to save plot (if None, don't save)
            show: Whether to display plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        policy_names = list(results.keys())
        colors = plt.cm.Set1(jnp.linspace(0, 1, len(policy_names)))
        
        # 1. Reward distributions
        ax = axes[0, 0]
        for i, name in enumerate(policy_names):
            rewards = results[name].total_rewards
            ax.hist(rewards, alpha=0.6, label=name, color=colors[i], bins=20)
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Box plot of rewards
        ax = axes[0, 1]
        reward_data = [results[name].total_rewards for name in policy_names]
        bp = ax.boxplot(reward_data, labels=policy_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Total Reward')
        ax.set_title('Reward Comparison')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 3. Final inventory distributions
        ax = axes[1, 0]
        for i, name in enumerate(policy_names):
            inventories = jnp.abs(results[name].final_inventories)
            ax.hist(inventories, alpha=0.6, label=name, color=colors[i], bins=20)
        ax.set_xlabel('Final Inventory (Absolute)')
        ax.set_ylabel('Frequency')
        ax.set_title('Final Inventory Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Performance metrics comparison
        ax = axes[1, 1]
        metrics = ['mean_reward', 'sharpe_ratio', 'liquidation_rate']
        x = jnp.arange(len(metrics))
        width = 0.8 / len(policy_names)
        
        for i, name in enumerate(policy_names):
            values = [results[name].metrics[metric] for metric in metrics]
            ax.bar(x + i * width, values, width, label=name, 
                  color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics')
        ax.set_xticks(x + width * (len(policy_names) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_sample_trajectories(self, results: Dict[str, PolicyResult],
                                n_samples: int = 3, save_path: str = None, 
                                show: bool = True):
        """
        Plot sample trajectories from each policy.
        
        Args:
            results: Results from compare_policies
            n_samples: Number of sample trajectories per policy
            save_path: Path to save plot
            show: Whether to display plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        policy_names = list(results.keys())
        colors = plt.cm.Set1(jnp.linspace(0, 1, len(policy_names)))
        
        variables = ['Price (Y)', 'Inventory (X)', 'Belief (p)', 'Action (u)']
        indices = [0, 1, 2, None]  # State indices, None for action
        
        for var_idx, (var_name, state_idx) in enumerate(zip(variables, indices)):
            ax = axes[var_idx // 2, var_idx % 2]
            
            for policy_idx, name in enumerate(policy_names):
                trajectories = results[name].trajectories[:n_samples]
                
                for traj_idx, traj in enumerate(trajectories):
                    if state_idx is not None:
                        # State variable
                        data = traj['states'][:, state_idx]
                    else:
                        # Action variable
                        data = traj['actions']
                    
                    times = self.config.time_grid[:len(data)]
                    alpha = 0.7 if traj_idx == 0 else 0.3
                    label = name if traj_idx == 0 else None
                    
                    ax.plot(times, data, color=colors[policy_idx], 
                           alpha=alpha, label=label)
            
            ax.set_xlabel('Time')
            ax.set_ylabel(var_name)
            ax.set_title(f'{var_name} Trajectories')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def generate_report(self, results: Dict[str, PolicyResult], 
                       stats: Dict[str, Any]) -> str:
        """
        Generate text report of comparison results.
        
        Args:
            results: Results from compare_policies
            stats: Results from statistical_comparison
            
        Returns:
            Formatted text report
        """
        report = "OPTIMAL EXECUTION POLICY COMPARISON REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Policy ranking
        report += "POLICY RANKING (by mean reward):\n"
        for i, (name, reward) in enumerate(stats['ranking']):
            report += f"{i+1}. {name}: {reward:.4f}\n"
        report += "\n"
        
        # Detailed metrics
        report += "DETAILED PERFORMANCE METRICS:\n"
        for name, result in results.items():
            report += f"\n{name}:\n"
            report += f"  Mean Reward: {result.metrics['mean_reward']:.4f} ± {result.metrics['std_reward']:.4f}\n"
            report += f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.4f}\n"
            report += f"  Liquidation Rate: {result.metrics['liquidation_rate']:.2%}\n"
            report += f"  Mean Final Inventory: {result.metrics['mean_final_inventory']:.4f}\n"
        
        # Statistical significance
        report += "\nSTATISTICAL COMPARISONS:\n"
        for comparison, data in stats['pairwise_comparisons'].items():
            significance = "**" if data['significant'] else ""
            report += f"{comparison}: "
            report += f"Δ={data['mean_difference']:.4f}, "
            report += f"t={data['t_statistic']:.2f} {significance}\n"
        
        return report
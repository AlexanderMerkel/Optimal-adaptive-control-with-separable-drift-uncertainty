"""
Unified Benchmark Evaluation Framework

This script provides a comprehensive comparison between different control strategies:
1. REINFORCE (adaptive learning with regime detection)
2. Certainty Equivalent (CE) control with belief-based expected parameters
3. Neumann-Voß 2022 optimal execution with mean regime parameters

The framework generates statistical comparisons, performance metrics, and visualizations.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from jax import random

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "rl" / "reinforce"))

from comparisons.certainty_equivalent_jax import CertaintyEquivalentController
from comparisons.neumann_voss_2022 import NeumannVoss2022Controller

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_config


class BenchmarkEvaluator:
    """Unified evaluation framework for all control benchmarks."""

    def __init__(self, config=None, reinforce_params=None, reinforce_apply_fn=None):
        """Initialize with centralized configuration and optional REINFORCE policy for comparison."""
        self.config = config if config is not None else get_config()
        self.T = self.config.T
        self.N = 200  # Default evaluation steps
        self.ce_controller = CertaintyEquivalentController(self.config)
        self.neumann_voss_controller = NeumannVoss2022Controller(self.config)
        self.reinforce_params = reinforce_params
        self.reinforce_apply_fn = reinforce_apply_fn

    def evaluate_all_methods(self, key, num_trajectories=100, n_steps=200, include_reinforce=True):
        """Evaluate all available control methods."""
        results = {}
        if include_reinforce:
            keys = random.split(key, 3)  # CE + Neumann-Voß + REINFORCE
        else:
            keys = random.split(key, 2)  # CE + Neumann-Voß only

        print("Evaluating Certainty Equivalent control...")
        results['ce'] = self.ce_controller.evaluate_performance(
            keys[0], num_trajectories, n_steps
        )

        print("Evaluating Neumann-Voß 2022 control...")
        results['neumann_voss'] = self.neumann_voss_controller.evaluate_performance(
            keys[1], num_trajectories, n_steps
        )

        if include_reinforce:
            print("Evaluating REINFORCE control...")
            results['reinforce'] = self._evaluate_reinforce(
                keys[2], num_trajectories, n_steps
            )

        return results

    def _evaluate_reinforce(self, key, num_trajectories=100, n_steps=200):
        """Evaluate REINFORCE policy with integrated training."""
        from rl.reinforce.reinforce_controller_jax import REINFORCEController
        
        print("  Creating and training REINFORCE controller...")
        controller = REINFORCEController(self.config)
        
        # Quick training for fair comparison
        controller.train_policy(num_episodes=200, verbose=False)
        
        print("  Evaluating trained REINFORCE policy...")
        return controller.evaluate_performance(key, num_trajectories, n_steps)

    def statistical_comparison(self, results):
        """Perform statistical comparison between methods."""
        methods = list(results.keys())
        profits = {method: results[method]['total_profits'] for method in methods
                  if len(results[method]['total_profits']) > 0}

        comparison = {}

        # Basic statistics
        comparison['summary'] = {}
        for method in profits:
            comparison['summary'][method] = {
                'mean': np.mean(profits[method]),
                'std': np.std(profits[method]),
                'min': np.min(profits[method]),
                'max': np.max(profits[method]),
                'median': np.median(profits[method]),
                'q25': np.percentile(profits[method], 25),
                'q75': np.percentile(profits[method], 75)
            }

        # Pairwise comparisons (simplified t-test-like comparison)
        comparison['pairwise'] = {}
        method_list = list(profits.keys())

        for i, method1 in enumerate(method_list):
            for method2 in method_list[i+1:]:
                profit1, profit2 = profits[method1], profits[method2]

                # Difference in means
                mean_diff = np.mean(profit1) - np.mean(profit2)

                # Pooled standard error (simplified)
                se1, se2 = np.std(profit1) / np.sqrt(len(profit1)), np.std(profit2) / np.sqrt(len(profit2))
                pooled_se = np.sqrt(se1**2 + se2**2)

                # Simple z-score
                z_score = mean_diff / pooled_se if pooled_se > 0 else 0

                comparison['pairwise'][f'{method1}_vs_{method2}'] = {
                    'mean_difference': mean_diff,
                    'pooled_se': pooled_se,
                    'z_score': z_score,
                    'significant': abs(z_score) > 1.96  # Approximate 95% confidence
                }

        return comparison

    def regime_performance_analysis(self, results):
        """Analyze performance by true regime for each method."""
        regime_analysis = {}

        for method_name, result in results.items():
            if len(result['total_profits']) == 0:
                continue

            true_regimes = result['true_regimes']
            profits = result['total_profits']

            # Split by true regime
            low_mask = (true_regimes == 0)
            high_mask = (true_regimes == 1)

            regime_analysis[method_name] = {
                'low_regime': {
                    'count': int(np.sum(low_mask)),
                    'mean_profit': float(np.mean(profits[low_mask])) if np.any(low_mask) else 0.0,
                    'std_profit': float(np.std(profits[low_mask])) if np.any(low_mask) else 0.0
                },
                'high_regime': {
                    'count': int(np.sum(high_mask)),
                    'mean_profit': float(np.mean(profits[high_mask])) if np.any(high_mask) else 0.0,
                    'std_profit': float(np.std(profits[high_mask])) if np.any(high_mask) else 0.0
                }
            }

        return regime_analysis

    def create_comparison_plots(self, results, save_path=None):
        """Create comprehensive comparison plots."""
        # Filter out empty results
        valid_results = {k: v for k, v in results.items()
                        if len(v['total_profits']) > 0}

        if len(valid_results) < 2:
            print("Not enough valid results for comparison plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Profit Distribution Comparison
        ax1 = axes[0, 0]
        profits_list = []
        labels = []
        for method, result in valid_results.items():
            profits_list.append(result['total_profits'])
            labels.append(result['method'])

        ax1.boxplot(profits_list, labels=labels)
        ax1.set_title('Profit Distribution Comparison')
        ax1.set_ylabel('Total Profit')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 2. Mean Performance Bar Chart
        ax2 = axes[0, 1]
        methods = list(valid_results.keys())
        means = [valid_results[m]['mean_profit'] for m in methods]
        stds = [valid_results[m]['std_profit'] for m in methods]

        x_pos = np.arange(len(methods))
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Mean Profit')
        ax2.set_title('Mean Performance Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([valid_results[m]['method'] for m in methods], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std/2,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        # 3. Regime Detection Accuracy
        ax3 = axes[1, 0]
        accuracies = [valid_results[m]['regime_accuracy'] for m in methods]
        bars = ax3.bar(x_pos, accuracies, alpha=0.7, color='lightcoral')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Regime Detection Accuracy')
        ax3.set_title('Regime Detection Performance')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([valid_results[m]['method'] for m in methods], rotation=45)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

        # 4. Performance by True Regime
        ax4 = axes[1, 1]
        regime_analysis = self.regime_performance_analysis(valid_results)

        x = np.arange(len(methods))
        width = 0.35

        low_means = [regime_analysis[m]['low_regime']['mean_profit'] for m in methods]
        high_means = [regime_analysis[m]['high_regime']['mean_profit'] for m in methods]

        ax4.bar(x - width/2, low_means, width, label='True Low Regime', alpha=0.7)
        ax4.bar(x + width/2, high_means, width, label='True High Regime', alpha=0.7)

        ax4.set_xlabel('Method')
        ax4.set_ylabel('Mean Profit')
        ax4.set_title('Performance by True Regime')
        ax4.set_xticks(x)
        ax4.set_xticklabels([valid_results[m]['method'] for m in methods], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to {save_path}")
        else:
            plt.show()

        return fig

    def generate_report(self, results, comparison_stats, regime_analysis, save_path=None):
        """Generate a comprehensive text report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("OPTIMAL EXECUTION BENCHMARK COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary statistics
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)

        # Filter valid results
        valid_results = {k: v for k, v in results.items()
                        if len(v['total_profits']) > 0}

        for method_name, result in valid_results.items():
            stats = comparison_stats['summary'][method_name]
            report_lines.append(f"{result['method']}:")
            report_lines.append(f"  Mean Profit: {stats['mean']:.4f} ± {stats['std']:.4f}")
            report_lines.append(f"  Median: {stats['median']:.4f}")
            report_lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            report_lines.append(f"  IQR: [{stats['q25']:.4f}, {stats['q75']:.4f}]")
            report_lines.append(f"  Regime Accuracy: {result['regime_accuracy']:.1%}")
            report_lines.append("")

        # Performance ranking
        ranked = sorted(valid_results.items(),
                       key=lambda x: x[1]['mean_profit'], reverse=True)

        report_lines.append("PERFORMANCE RANKING (by mean profit)")
        report_lines.append("-" * 40)
        for i, (method_name, result) in enumerate(ranked, 1):
            report_lines.append(f"{i}. {result['method']}: {result['mean_profit']:.4f}")
        report_lines.append("")

        # Pairwise comparisons
        report_lines.append("PAIRWISE STATISTICAL COMPARISONS")
        report_lines.append("-" * 40)
        for comparison, stats in comparison_stats['pairwise'].items():
            significance = "***" if stats['significant'] else ""
            report_lines.append(
                f"{comparison}: Δ={stats['mean_difference']:.4f}, "
                f"z={stats['z_score']:.2f} {significance}"
            )
        report_lines.append("")

        # Regime-specific performance
        report_lines.append("PERFORMANCE BY TRUE REGIME")
        report_lines.append("-" * 40)
        for method_name in valid_results.keys():
            if method_name in regime_analysis:
                analysis = regime_analysis[method_name]
                report_lines.append(f"{valid_results[method_name]['method']}:")
                report_lines.append(
                    f"  Low Regime:  {analysis['low_regime']['mean_profit']:.4f} "
                    f"± {analysis['low_regime']['std_profit']:.4f} "
                    f"(n={analysis['low_regime']['count']})"
                )
                report_lines.append(
                    f"  High Regime: {analysis['high_regime']['mean_profit']:.4f} "
                    f"± {analysis['high_regime']['std_profit']:.4f} "
                    f"(n={analysis['high_regime']['count']})"
                )
                report_lines.append("")

        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")

        return report_text

    def run_full_evaluation(self, key, num_trajectories=100, n_steps=200,
                           save_plots=True, save_report=True, output_dir="outputs"):
        """Run complete benchmark evaluation with all analyses."""
        print("Starting comprehensive benchmark evaluation...")
        print(f"Trajectories per method: {num_trajectories}")
        print(f"Time steps: {n_steps}")
        print("-" * 50)

        # Evaluate all methods
        results = self.evaluate_all_methods(key, num_trajectories, n_steps)

        # Statistical analysis
        print("Performing statistical analysis...")
        comparison_stats = self.statistical_comparison(results)
        regime_analysis = self.regime_performance_analysis(results)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate plots
        if save_plots:
            print("Creating comparison plots...")
            plot_path = output_path / "benchmark_comparison.png"
            self.create_comparison_plots(results, plot_path)

        # Generate report
        if save_report:
            print("Generating report...")
            report_path = output_path / "benchmark_report.txt"
            report_text = self.generate_report(results, comparison_stats, regime_analysis, report_path)
            print("\nREPORT PREVIEW:")
            print(report_text[:1000] + "..." if len(report_text) > 1000 else report_text)

        print("\nEvaluation complete!")

        return {
            'results': results,
            'comparison_stats': comparison_stats,
            'regime_analysis': regime_analysis
        }


if __name__ == "__main__":
    # Run comprehensive benchmark evaluation
    key = random.PRNGKey(42)
    evaluator = BenchmarkEvaluator()

    # Run evaluation with moderate number of trajectories for testing
    evaluation = evaluator.run_full_evaluation(
        key,
        num_trajectories=100,
        n_steps=200,
        save_plots=True,
        save_report=True
    )

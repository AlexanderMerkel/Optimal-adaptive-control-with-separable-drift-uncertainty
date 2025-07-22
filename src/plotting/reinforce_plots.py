import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from jax import random
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from rl.reinforce.reinforce_jax import (
    PolicyNetwork,
    reset_env,
    rollout,
    N,
    T,
)


def load_policy(params_path):
    """Load a saved policy from a pickle file"""
    with open(params_path, "rb") as f:
        numpy_params = pickle.load(f)

    params = jax.tree_util.tree_map(jnp.array, numpy_params)
    return params


def create_policy_function(hidden_dim=16):
    """Create a policy network and its apply function"""
    policy = PolicyNetwork(hidden_dim=hidden_dim)
    return policy


def plot_control_trajectories(
    policy_params_path,
    num_paths=10,
    n_steps=N,
    save_path=None,
    seed=42,
    hidden_dim=16,
    show_plots=False,
):
    """
    Plot control and inventory trajectories using a saved policy

    Args:
        policy_params_path: Path to the saved policy parameters
        num_paths: Number of trajectories to plot
        n_steps: Number of steps in each trajectory
        save_path: Path to save the plots (if None, plots are displayed)
        seed: Random seed
        hidden_dim: Hidden dimension of the policy network
        show_plots: Whether to display the plots
    """
    print(f"Loading policy from {policy_params_path}")
    key = random.PRNGKey(seed)

    policy_params = load_policy(policy_params_path)

    policy = create_policy_function(hidden_dim)
    policy_apply = policy.apply

    print(f"Generating {num_paths} trajectories...")

    key, reset_key, rollout_key = random.split(key, 3)
    initial_state = reset_env(reset_key, batch_size=num_paths)

    states, actions, _, _, _ = rollout(
        rollout_key, policy_params, policy_apply, initial_state, n_steps, num_paths
    )

    print("Converting to numpy arrays...")
    actions_np = np.array(actions)
    states_np = np.array(states)
    time_steps = np.linspace(0, T, n_steps + 1)[:-1]

    plt.close("all")

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    print("Plotting control actions...")

    plt.figure(figsize=(10, 6))
    for i in range(num_paths):
        plt.plot(time_steps, actions_np[:, i], alpha=0.7)

    plt.title(f"Optimal Control Policy (Actions) - {num_paths} Sample Paths")
    plt.xlabel("Time (t)")
    plt.ylabel("Action")
    plt.grid(True)

    if save_path:
        plot_filepath = os.path.join(save_path, "optimal_control_paths.png")
        plt.savefig(plot_filepath)
        print(f"Control plot saved to {plot_filepath}")

    plt.close()

    print("Plotting inventory paths...")

    plt.figure(figsize=(10, 6))
    for i in range(num_paths):
        plt.plot(time_steps, states_np[:, i, 2], alpha=0.7)

    plt.title(f"Optimal Inventory Paths (X) - {num_paths} Sample Paths")
    plt.xlabel("Time (t)")
    plt.ylabel("Inventory (X)")
    plt.grid(True)

    if save_path:
        plot_filepath = os.path.join(save_path, "optimal_inventory_paths.png")
        plt.savefig(plot_filepath)
        print(f"Inventory plot saved to {plot_filepath}")

    plt.close()

    print("Plotting complete")
    return states_np, actions_np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot trajectories from a trained policy")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to the saved policy parameters"
    )
    parser.add_argument("--num_paths", type=int, default=10, help="Number of trajectories to plot")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--hidden_dim", type=int, default=16, help="Hidden dimension of the policy network"
    )

    args = parser.parse_args()

    try:
        plot_control_trajectories(
            args.policy_path,
            num_paths=args.num_paths,
            save_path=args.save_path,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
        )
        print("Script completed successfully")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()

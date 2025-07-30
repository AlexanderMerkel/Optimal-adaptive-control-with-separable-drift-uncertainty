#!/usr/bin/env python3
"""Debug trajectory generation issue."""

import jax
import jax.numpy as jnp
from jax import random

from src.control_theory import OptimalExecutionEnvironment, RiccatiPolicy, TrajectoryGenerator
from src.utils import get_config, RiccatiSolver

def main():
    print("Debugging trajectory generation...")
    
    config = get_config()
    key = random.PRNGKey(42)
    
    # Create environment
    environment = OptimalExecutionEnvironment(config)
    print(f"Environment created: {environment}")
    
    # Test environment reset
    key, reset_key = random.split(key)
    initial_state = environment.reset(reset_key, batch_size=2)
    print(f"Initial state shape: {initial_state.data.shape}")
    print(f"Initial state time: {initial_state.time}")
    print(f"Initial state metadata: {initial_state.metadata}")
    
    if initial_state.metadata:
        for k, v in initial_state.metadata.items():
            print(f"  {k}: shape={getattr(v, 'shape', 'scalar')}, dtype={getattr(v, 'dtype', type(v))}")
    
    # Create policy
    riccati_solver = RiccatiSolver(config)
    lambda_mean = 0.5 * (config.LAMBDA_L + config.LAMBDA_H)
    
    policy = RiccatiPolicy(
        riccati_solver=riccati_solver,
        lambda_func=lambda_mean,
        rho=config.RHO,
        state_indices={'X': 2}
    )
    print(f"Policy created: {policy}")
    
    # Test policy action
    key, action_key = random.split(key)
    action = policy.compute_action(initial_state, time=0.0, key=action_key)
    print(f"Action shape: {action.data.shape}")
    print(f"Action metadata: {action.metadata}")
    
    # Create trajectory generator
    generator = TrajectoryGenerator(policy, environment, compile_trajectory_gen=False)
    print(f"Trajectory generator created: {generator}")
    
    # Test single trajectory
    key, traj_key = random.split(key)
    single_state = environment.reset(traj_key, batch_size=1)
    print(f"Single state shape: {single_state.data.shape}")
    print(f"Single state metadata: {single_state.metadata}")
    
    try:
        trajectory = generator.generate_trajectory(single_state, n_steps=5, key=traj_key)
        print(f"Single trajectory generated successfully!")
        print(f"  States shape: {trajectory.states.shape}")
        print(f"  Actions shape: {trajectory.actions.shape}")
        print(f"  Rewards shape: {trajectory.rewards.shape}")
    except Exception as e:
        print(f"Single trajectory failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test batch trajectory
    try:
        key, batch_key = random.split(key)
        batch_trajectories = generator.generate_batch_trajectories(
            batch_size=2, n_steps=5, key=batch_key
        )
        print(f"Batch trajectories generated successfully!")
        print(f"  States shape: {batch_trajectories.states.shape}")
        print(f"  Actions shape: {batch_trajectories.actions.shape}")
        print(f"  Rewards shape: {batch_trajectories.rewards.shape}")
    except Exception as e:
        print(f"Batch trajectory failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
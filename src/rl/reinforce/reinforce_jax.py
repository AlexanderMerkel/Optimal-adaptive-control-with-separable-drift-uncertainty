import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
from jax import random
from functools import partial
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


yaml_path = Path(__file__).parent.parent.parent / "model_parameters.yaml"
with open(yaml_path, "r") as file:
    params = yaml.safe_load(file)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
MAX_ACTION_CLIP = 10.0
MAX_GRAD_NORM = 500.0
ZERO_GRAD_THRESHOLD = 1e-3
EARLY_STOPPING_PATIENCE = 300
BATCH_SIZE = 256
N = 200


T = float(params["T"])
dt = T / N
RHO = float(params["RHO"])
SIGMA = float(params["SIGMA"])
C_RUNNING = float(params["C_RUNNING"])
C_TERMINAL = float(params["C_TERMINAL"])
LAMBDA_L = float(params["LAMBDA_L"])
LAMBDA_H = float(params["LAMBDA_H"])
KAPPA_L = float(params["KAPPA_L"])
KAPPA_H = float(params["KAPPA_H"])
STATE_BOUNDS = params["STATE_BOUNDS"]
INITIAL_STATE = params["INITIAL_STATE"]


low_bounds = jnp.array([v[0] for v in STATE_BOUNDS.values()], dtype=jnp.float32)
high_bounds = jnp.array([v[1] for v in STATE_BOUNDS.values()], dtype=jnp.float32)


class PolicyNetwork(nn.Module):
    hidden_dim: int = 128
    action_dim: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.tanh(x)
        mean = nn.Dense(features=self.action_dim)(x)
        log_std = nn.Dense(features=self.action_dim)(x)

        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std


@partial(jax.jit, static_argnames=["batch_size"])
def reset_env(key: random.PRNGKey, batch_size: int = BATCH_SIZE):
    initial_values = jnp.array(
        [INITIAL_STATE[k] for k in ["t", "S", "X", "p", "A_l", "A_h"]], dtype=jnp.float32
    )
    state = jnp.tile(initial_values, (batch_size, 1))
    return state


@partial(jax.jit, static_argnames=["batch_size"])
def step_env(
    key: random.PRNGKey, state: jnp.ndarray, actions: jnp.ndarray, batch_size: int = BATCH_SIZE
):
    key, subkey = random.split(key)
    dW_batch = jnp.sqrt(dt) * random.normal(subkey, (batch_size,), dtype=jnp.float32)

    t, S, X, p, A_l, A_h = state.T
    actions = actions.reshape(batch_size)

    driftS = -(
        LAMBDA_L * (actions + KAPPA_L * A_l) * p + LAMBDA_H * (actions + KAPPA_H * A_h) * (1.0 - p)
    )
    dS = driftS * dt + dW_batch
    dX = -actions * dt
    dp = (
        p
        * (1.0 - p)
        * (LAMBDA_L * (actions + KAPPA_L * A_l) - LAMBDA_H * (actions + KAPPA_H * A_h))
        * dW_batch
    )

    dA_l = (actions + KAPPA_L * A_l) * dt
    dA_h = (actions + KAPPA_H * A_h) * dt

    t_next = t + dt
    S_next = S + dS
    X_next = X + dX
    p_next = p + dp
    A_l_next = A_l + dA_l
    A_h_next = A_h + dA_h

    next_state_unclamped = jnp.stack([t_next, S_next, X_next, p_next, A_l_next, A_h_next], axis=1)
    next_state = jnp.clip(next_state_unclamped, low_bounds, high_bounds)

    S_batch, X_batch = next_state[:, 1], next_state[:, 2]

    instant_profit = ((S_batch - RHO * actions) * actions - C_RUNNING * (X_batch**2)) * dt
    done = jnp.zeros(batch_size, dtype=bool)

    return next_state, instant_profit, done


@partial(jax.jit, static_argnames=["policy_apply", "n_steps", "batch_size"])
def rollout(
    key: random.PRNGKey,
    policy_params: dict,
    policy_apply: callable,
    initial_state: jnp.ndarray,
    n_steps: int = N,
    batch_size: int = BATCH_SIZE,
):
    def scan_body(carry, _):
        key, current_state = carry
        key, policy_key, env_key = random.split(key, 3)

        mean, log_std = policy_apply({"params": policy_params}, current_state)
        mean = jnp.squeeze(mean, axis=-1)
        log_std = jnp.squeeze(log_std, axis=-1)

        pi = distrax.Normal(loc=mean, scale=jnp.exp(log_std))
        actions_unclipped = pi.sample(seed=policy_key)

        # Apply tanh squashing
        actions_tanh = jnp.tanh(actions_unclipped)
        actions = actions_tanh * MAX_ACTION_CLIP

        # Calculate log_prob using change of variables formula for tanh
        log_probs = pi.log_prob(actions_unclipped) - jnp.log(
            MAX_ACTION_CLIP * (1 - actions_tanh**2) + 1e-6
        )

        next_state, reward, _ = step_env(env_key, current_state, actions, batch_size)

        new_carry = (key, next_state)

        # Return squashed actions and corrected log_probs
        outputs = (current_state, actions, reward, log_probs)
        return new_carry, outputs

    key, initial_env_key = random.split(key)
    initial_carry = (initial_env_key, initial_state)
    (_, final_state), (states, actions, rewards, log_probs) = jax.lax.scan(
        scan_body, initial_carry, None, length=n_steps
    )

    S_final, X_final = final_state[:, 1], final_state[:, 2]

    terminal_profit = S_final * X_final - C_TERMINAL * X_final**2

    rewards = rewards.at[-1, :].add(terminal_profit)

    return states, actions, rewards, log_probs, final_state


@partial(jax.jit, static_argnames=["policy_apply", "optimizer_update", "n_steps", "batch_size"])
def train_step(
    key: random.PRNGKey,
    policy_params: dict,
    opt_state: optax.OptState,
    policy_apply: callable,
    optimizer_update: callable,
    n_steps: int = N,
    batch_size: int = BATCH_SIZE,
):
    key, rollout_key, reset_key = random.split(key, 3)
    initial_state = reset_env(reset_key, batch_size)

    def loss_fn(params):
        _states, _actions, rewards, log_probs, _final_state = rollout(
            rollout_key, params, policy_apply, initial_state, n_steps, batch_size
        )

        total_profit = jnp.sum(rewards, axis=0)
        # Sum log_probs over time for each trajectory
        sum_log_probs_per_traj = jnp.sum(log_probs, axis=0)

        safe_total_profit = jnp.nan_to_num(total_profit, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)

        # Correct loss calculation: use sum of log_probs per trajectory
        loss = -jnp.mean(sum_log_probs_per_traj * jax.lax.stop_gradient(safe_total_profit))

        loss = jnp.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)

        return loss, total_profit

    (loss, total_profit_batch), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_params)

    updates, new_opt_state = optimizer_update(grads, opt_state, policy_params)
    new_policy_params = optax.apply_updates(policy_params, updates)

    metrics = {"loss": loss, "mean_total_profit": jnp.mean(total_profit_batch)}

    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    metrics["grad_norm"] = jnp.linalg.norm(jnp.concatenate([jnp.ravel(g) for g in grads_flat]))

    return new_policy_params, new_opt_state, metrics


def train_policy_jax(num_episodes=5000, batch_size=BATCH_SIZE, learning_rate=1e-4, seed=42):
    key = random.PRNGKey(seed)
    policy = PolicyNetwork()
    dummy_state = jnp.zeros((1, 6), dtype=jnp.float32)
    key, init_key = random.split(key)
    policy_params = policy.init(init_key, dummy_state)["params"]

    optimizer = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(learning_rate))

    opt_state = optimizer.init(policy_params)

    policy_apply_fn = policy.apply
    optimizer_update_fn = optimizer.update

    jitted_train_step = partial(
        train_step,
        policy_apply=policy_apply_fn,
        optimizer_update=optimizer_update_fn,
        n_steps=N,
        batch_size=batch_size,
    )

    zero_grad_counter = 0
    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        key, step_key = random.split(key)
        policy_params, opt_state, metrics = jitted_train_step(step_key, policy_params, opt_state)

        if episode % 100 == 0:
            print(
                f"Episode {episode}, Loss: {metrics['loss']:.4f}, "
                f"Mean Profit: {metrics['mean_total_profit']:.4f}, "
                f"Grad Norm: {metrics['grad_norm']:.4f}"
            )

        if jnp.isnan(metrics["loss"]) or jnp.isinf(metrics["loss"]):
            print(f"NaN or Inf detected in loss at episode {episode}. Stopping.")
            break

        if jnp.isnan(metrics["mean_total_profit"]) or jnp.isinf(metrics["mean_total_profit"]):
            print(f"NaN or Inf detected in profit at episode {episode}. Stopping.")
            break

        if metrics["grad_norm"] < ZERO_GRAD_THRESHOLD:
            zero_grad_counter += 1
        else:
            zero_grad_counter = 0

        if zero_grad_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"Gradient norm has been below {ZERO_GRAD_THRESHOLD} for {EARLY_STOPPING_PATIENCE} consecutive steps. Stopping early at episode {episode}."
            )
            break

    print("Training finished.")
    return policy_params, policy_apply_fn


def save_policy_jax(params, path):
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, "policy_params.pkl")

    numpy_params = jax.tree_util.tree_map(np.array, params)
    with open(filepath, "wb") as f:
        pickle.dump(numpy_params, f)
    print(f"Policy parameters saved to {filepath}")


def simulate_and_plot(key, policy_params, policy_apply, num_paths=10, n_steps=N, save_path=None):
    key, reset_key, rollout_key = random.split(key, 3)
    initial_state = reset_env(reset_key, batch_size=num_paths)

    states, actions, _, _, _ = rollout(
        rollout_key, policy_params, policy_apply, initial_state, n_steps, num_paths
    )

    actions_np = np.array(actions)
    states_np = np.array(states)
    time_steps = np.linspace(0, T, n_steps + 1)[:-1]

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
        print(f"Plot saved to {plot_filepath}")
    else:
        plt.show()

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
    else:
        plt.show()


if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.abspath(".")

    save_dir = os.path.abspath(os.path.join(script_dir, "../../../outputs/reinforce_jax_stable"))

    main_key = random.PRNGKey(42)
    train_key, plot_key = random.split(main_key)

    trained_params, policy_apply_fn = train_policy_jax(
        num_episodes=500000, batch_size=BATCH_SIZE, learning_rate=1e-4, seed=int(train_key[0])
    )

    if trained_params:
        save_policy_jax(trained_params, save_dir)
        simulate_and_plot(
            plot_key, trained_params, policy_apply_fn, num_paths=10, save_path=save_dir
        )

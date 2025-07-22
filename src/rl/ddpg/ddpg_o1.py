import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from functools import partial
import random

# ---------------------------------------------
# Hyperparameters & Model Constants
# ---------------------------------------------
DT = 0.01              # Euler-Maruyama time step
GAMMA = 0.99           # Discount factor
TAU = 0.005            # Soft-update rate for target nets
REPLAY_SIZE = 10_000   # Max size for replay buffer
BATCH_SIZE = 64        # Training batch size

# Problem-specific constants
lambda_l, lambda_h = 1.0, 1.5
kappa_l, kappa_h = 0.2, 0.3
rho, c = 0.1, 0.05

# ---------------------------------------------
# 1. State Dynamics and Reward
# ---------------------------------------------
def step_state_single(x_packed, u, dW):
    """
    Single-state Euler-Maruyama step.

    Args:
      x_packed: (5,) state vector [S, X, p, A_l, A_h].
      u: (,) action (scalar).
      dW: (,) Brownian increment (scalar).

    Returns:
      next_x: (5,) next state.
    """
    S, Xval, p, A_l, A_h = x_packed

    # drift term for S
    driftS = (lambda_l * (u + kappa_l * A_l) * p
              + lambda_h * (u + kappa_h * A_h) * (1. - p))
    dS = driftS * DT + dW

    dX = -u * DT

    # diffusion in p multiplied by the Brownian increment
    dp = p * (1. - p) * (lambda_l * (u + kappa_l * A_l)
                         - lambda_h * (u + kappa_h * A_h)) * dW

    dA_l = (u - kappa_l * A_l) * DT
    dA_h = (u - kappa_h * A_h) * DT

    return jnp.array([S + dS, Xval + dX, p + dp, A_l + dA_l, A_h + dA_h])

# Vectorized version of the environment step over a batch of states:
step_state_batch = jax.vmap(step_state_single, in_axes=(0, 0, 0))

def reward_function_batch(states, actions):
    """
    Computes reward for each state-action pair in the batch.

    Args:
      states: (batch_size, 5) state batch.
      actions: (batch_size,) action batch (1D continuous).

    Returns:
      rewards: (batch_size,) reward values.
    """
    # For convenience, the user-specified cost:
    # instant_cost = ((P_batch - rho * u_batch) * u_batch - c * (X_batch**2)) * DT
    P_batch = states[:, 0]
    X_batch = states[:, 1]
    return ((P_batch - rho * actions) * actions - c * (X_batch**2)) * DT

# ---------------------------------------------
# 2. Network Definitions (Actor & Critic)
# ---------------------------------------------
def actor_mlp_fn(state_dim=5, hidden_sizes=(64, 64)):
    """MLP mapping R^5 -> R^1 for the actor."""
    # Haiku forward pass
    mlp = hk.Sequential([
        hk.Linear(hidden_sizes[0]), jax.nn.relu,
        hk.Linear(hidden_sizes[1]), jax.nn.relu,
        hk.Linear(1)  # Output dimension is 1 (deterministic action)
    ])
    return mlp

def critic_mlp_fn(state_dim=5, action_dim=1, hidden_sizes=(64, 64)):
    """MLP mapping R^(5+1) -> R^1 for the critic."""
    mlp = hk.Sequential([
        hk.Linear(hidden_sizes[0]), jax.nn.relu,
        hk.Linear(hidden_sizes[1]), jax.nn.relu,
        hk.Linear(1)  # Scalar Q-value
    ])
    return mlp

def build_actor():
    """Haiku transform for actor network."""
    def forward_actor(states):
        mlp = actor_mlp_fn()
        return mlp(states)
    return hk.transform(forward_actor)

def build_critic():
    """Haiku transform for critic network."""
    def forward_critic(states, actions):
        # Concatenate states & actions along last dimension
        inp = jnp.concatenate([states, actions[..., None]], axis=-1)
        mlp = critic_mlp_fn()
        return mlp(inp)
    return hk.transform(forward_critic)

# ---------------------------------------------
# 3. Replay Buffer
# ---------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, s, a, r, ns, d):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, ns, d))
    
    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (jnp.array(s),
                jnp.array(a),
                jnp.array(r),
                jnp.array(ns),
                jnp.array(d))

# ---------------------------------------------
# 4. Initialization Helpers
# ---------------------------------------------
def init_networks(rng_key, batch_size=1):
    """
    Initializes the actor and critic networks (including target).
    
    Args:
      rng_key: JAX random key.
      batch_size: shape used for dummy initialization.
    Returns:
      actor_params, actor_apply_fn, critic_params, critic_apply_fn,
      actor_target_params, critic_target_params
    """
    actor = build_actor()
    critic = build_critic()

    # Dummy inputs for shape inference
    dummy_state = jnp.zeros((batch_size, 5))
    dummy_action = jnp.zeros((batch_size,))

    # Init
    actor_params = actor.init(rng_key, dummy_state)
    critic_params = critic.init(rng_key, dummy_state, dummy_action)

    # Make copies for target networks
    actor_target_params = jax.tree_util.tree_map(lambda x: x.copy(), actor_params)
    critic_target_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
    
    return (actor_params, actor.apply, critic_params, critic.apply,
            actor_target_params, critic_target_params)

# ---------------------------------------------
# 5. Optimizers and Soft Updates
# ---------------------------------------------
def soft_update(main_params, target_params, tau=TAU):
    return jax.tree_util.tree_map(lambda mp, tp: tau * mp + (1 - tau) * tp,
                                  main_params, target_params)

def get_optimizers(lr=3e-4):
    actor_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    return actor_opt, critic_opt

# ---------------------------------------------
# 6. Training Step: DDPG Update
# ---------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def ddpg_update(batch_size,
                actor_params, actor_apply_fn,
                critic_params, critic_apply_fn,
                actor_target_params, critic_target_params,
                opt_actor_state, opt_critic_state,
                actor_opt, critic_opt,
                rng_key,
                states, actions, rewards, next_states, dones):
    """
    One gradient step for DDPG.
    """
    # ------------------
    # 1. Critic Update
    # ------------------
    def critic_loss_fn(critic_params_, actor_target_params_, critic_target_params_):
        # Target actions
        next_actions = actor_apply_fn(actor_target_params_, next_states)
        # Target Q
        next_q = critic_apply_fn(critic_target_params_, next_states, next_actions.squeeze(-1))
        target_q = rewards + GAMMA * (1. - dones) * next_q

        # Current Q
        q_vals = critic_apply_fn(critic_params_, states, actions)
        return jnp.mean((q_vals - target_q) ** 2)

    critic_grad_fn = jax.grad(critic_loss_fn)
    c_grads = critic_grad_fn(critic_params, actor_target_params, critic_target_params)
    updates_c, opt_critic_state = critic_opt.update(c_grads, opt_critic_state, critic_params)
    new_critic_params = optax.apply_updates(critic_params, updates_c)

    # ------------------
    # 2. Actor Update
    # ------------------
    def actor_loss_fn(actor_params_, critic_params_):
        # Actor uses current states to produce actions
        current_actions = actor_apply_fn(actor_params_, states).squeeze(-1)
        q_vals = critic_apply_fn(critic_params_, states, current_actions)
        return -jnp.mean(q_vals)

    actor_grad_fn = jax.grad(actor_loss_fn)
    a_grads = actor_grad_fn(actor_params, new_critic_params)
    updates_a, opt_actor_state = actor_opt.update(a_grads, opt_actor_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates_a)

    # ------------------
    # 3. Soft Update Targets
    # ------------------
    new_actor_target_params = soft_update(new_actor_params, actor_target_params)
    new_critic_target_params = soft_update(new_critic_params, critic_target_params)

    return (new_actor_params, new_critic_params,
            new_actor_target_params, new_critic_target_params,
            opt_actor_state, opt_critic_state)

# ---------------------------------------------
# 7. Vectorized Environment Rollout
# ---------------------------------------------
def rollout_batch(rng_key, actor_apply_fn, actor_params,
                  states, num_steps):
    """
    Simulate a batch of trajectories for 'num_steps' using the current actor.
    
    Args:
      rng_key: JAX random key.
      actor_apply_fn: actor's apply function.
      actor_params: actor's parameters.
      states: (batch_size, 5) initial states.
      num_steps: how many transitions to simulate.

    Returns:
      transitions: list of (s, a, r, s', done) for each step in the batch.
    """
    batch_size = states.shape[0]
    
    def scan_fn(carry, _):
        rng_k, cur_states = carry
        # Deterministic action
        acts = actor_apply_fn(actor_params, cur_states).squeeze(-1)

        # Brownian increments for each state in the batch
        rng_k, sub_k = jax.random.split(rng_k)
        dW = jax.random.normal(sub_k, shape=(batch_size,)) * jnp.sqrt(DT)

        # Next state
        nxt_states = step_state_batch(cur_states, acts, dW)

        # Reward
        rews = reward_function_batch(cur_states, acts)

        # For this example, assume never "done" (set done=0).
        done = jnp.zeros((batch_size,))

        transition = (cur_states, acts, rews, nxt_states, done)
        return (rng_k, nxt_states), transition

    # Transform scan_fn to work with static arguments
    scan_fn = jax.vmap(lambda x, _: scan_fn(x, _), in_axes=(0, None))
    
    init_carry = (rng_key, states)
    _, transitions = jax.lax.scan(scan_fn, init_carry, None, length=num_steps)
    return transitions

# ---------------------------------------------
# 8. Main Training Loop (High-Level)
# ---------------------------------------------
def main_training(num_episodes=500, steps_per_episode=10, batch_size=BATCH_SIZE, seed=42):
    # Initialize replay buffer
    rb = ReplayBuffer(capacity=REPLAY_SIZE)

    rng = jax.random.PRNGKey(seed)
    # Initialize networks & optimizers
    (actor_params, actor_apply_fn,
     critic_params, critic_apply_fn,
     actor_tgt_params, critic_tgt_params) = init_networks(rng, batch_size=1)

    actor_opt, critic_opt = get_optimizers()
    opt_actor_state = actor_opt.init(actor_params)
    opt_critic_state = critic_opt.init(critic_params)

    # Initialize a batch of states for rollout:
    # Example: random initial states for the environment
    batch_size_env = 32  # number of parallel envs for data collection
    rng, subkey = jax.random.split(rng)
    init_states = jax.random.normal(subkey, shape=(batch_size_env, 5))

    # Populate replay buffer initially
    transitions = rollout_batch(rng, actor_apply_fn, actor_params, init_states, 100)
    # transitions is shape (100, batch_size_env, <each data>).
    # Flatten and store in replay buffer:
    for t in range(100):
        s, a, r, s_next, d = [x[t] for x in transitions]
        for i in range(batch_size_env):
            rb.add(np.array(s[i]), np.array(a[i]), np.array(r[i]),
                   np.array(s_next[i]), np.array(d[i]))

    # Training loop
    for ep in range(num_episodes):
        # Rollout: gather new transitions
        rng, subkey = jax.random.split(rng)
        init_states = jax.random.normal(subkey, shape=(batch_size_env, 5))
        rng, subkey = jax.random.split(rng)
        transitions = rollout_batch(subkey, actor_apply_fn, actor_params, init_states, steps_per_episode)
        
        # Store transitions in replay buffer
        for t in range(steps_per_episode):
            s, a, r, s_next, d = [x[t] for x in transitions]
            for i in range(batch_size_env):
                rb.add(np.array(s[i]), np.array(a[i]), np.array(r[i]),
                       np.array(s_next[i]), np.array(d[i]))

        # Sample random batch from replay buffer & train
        if len(rb.buffer) > batch_size:
            s_b, a_b, r_b, ns_b, d_b = rb.sample(batch_size)
            (actor_params, critic_params,
             actor_tgt_params, critic_tgt_params,
             opt_actor_state, opt_critic_state) = ddpg_update(
                batch_size,
                actor_params, actor_apply_fn,
                critic_params, critic_apply_fn,
                actor_tgt_params, critic_tgt_params,
                opt_actor_state, opt_critic_state,
                actor_opt, critic_opt,
                rng,
                s_b, a_b, r_b, ns_b, d_b
            )
        
        # Logging or additional metrics can be added here
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes} completed.")

    print("Training finished.")
    return actor_params, critic_params, actor_tgt_params, critic_tgt_params

# If you want to run training:
# actor_params, critic_params, actor_tgt_params, critic_tgt_params = main_training()

if __name__ == "__main__":
    # Run training
    actor_params, critic_params, actor_tgt_params, critic_tgt_params = main_training()
    print("Training complete.")
    print("Example: actor_params:", actor_params)
    print("Example: critic_params:", critic_params)
    print("Example: actor_tgt_params:", actor_tgt_params)
    print("Example: critic_tgt_params:", critic_tgt_params)
import jax
import jax.numpy as jnp
import optax
import random
from functools import partial
import flax.linen as nn

# Constants
dt = 0.01  # Time step
gamma = 0.99  # Discount factor
tau = 0.005  # Target network update rate
lambda_l, lambda_h = 1.0, 1.5  # Model-specific parameters
kappa_l, kappa_h = 0.2, 0.3
rho, c = 0.1, 0.05

# Dynamics: Euler-Maruyama
def step_state(x_packed, u, dW):
    """
    Args:
        x_packed: shape (5,) -> [S, X, p, A_l, A_h]
        u: shape () -> scalar action
        dW: shape () -> scalar Brownian increment
    Returns:
        shape (5,) -> updated state
    """
    S, Xval, p, A_l, A_h = x_packed
    
    # State transitions
    driftS = (lambda_l * (u + kappa_l * A_l) * p +
              lambda_h * (u + kappa_h * A_h) * (1. - p))
    dS = driftS * dt + dW
    
    dX = -u * dt
    dp = p * (1. - p) * (lambda_l * (u + kappa_l * A_l) -
                         lambda_h * (u + kappa_h * A_h)) * dW
    dA_l = (u - kappa_l * A_l) * dt
    dA_h = (u - kappa_h * A_h) * dt

    return jnp.array([S + dS, Xval + dX, p + dp, A_l + dA_l, A_h + dA_h])

# Reward computation
def compute_reward(state, action):
    """
    Args:
        state: shape (batch_size, 5)
        action: shape (batch_size, 1)
    Returns:
        shape (batch_size,)
    """
    P_batch, X_batch = state[:, 0], state[:, 1]
    return ((P_batch - rho * action.squeeze()) * action.squeeze() - c * (X_batch**2)) * dt

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = []
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(jnp.array, zip(*batch))

# Networks
def build_actor(state_dim):
    return optax.chain(
        optax.scale_by_adam(),
        optax.scale(-3e-4)
    )

def build_critic(state_dim, action_dim):
    return optax.chain(
        optax.scale_by_adam(),
        optax.scale(-3e-4)
    )

class Actor(nn.Module):
    """Actor network."""
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Ensure input is 2D
        x = x.reshape(-1, x.shape[-1])
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)

class Critic(nn.Module):
    """Critic network."""
    
    @nn.compact
    def __call__(self, state, action):
        # Ensure inputs have correct shape and concatenate along last dimension
        state = state.reshape(-1, state.shape[-1])
        action = action.reshape(-1, action.shape[-1])
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)

# Target Update
@jax.jit
def soft_update(params, target_params):
    return jax.tree_util.tree_map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)

# Training Step
@jax.jit
def train_step(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
               actor, critic, actor_params, critic_params, actor_target_params, critic_target_params,
               actor_opt_state, critic_opt_state, opt_actor, opt_critic, rng):
    
    # Compute target Q-values
    next_actions = actor.apply(actor_target_params, next_state_batch)
    next_q_values = critic.apply(critic_target_params, next_state_batch, next_actions)
    target_q = reward_batch + gamma * (1 - done_batch) * next_q_values.squeeze()
    
    # Update critic
    def critic_loss_fn(params):
        q_values = critic.apply(params, state_batch, action_batch)
        return jnp.mean((q_values.squeeze() - target_q) ** 2)
    
    critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
    updates, critic_opt_state = opt_critic.update(critic_grads, critic_opt_state)
    critic_params = optax.apply_updates(critic_params, updates)
    
    # Update actor
    def actor_loss_fn(params):
        actions = actor.apply(params, state_batch)
        q_values = critic.apply(critic_params, state_batch, actions)
        return -jnp.mean(q_values)
    
    actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
    updates, actor_opt_state = opt_actor.update(actor_grads, actor_opt_state)
    actor_params = optax.apply_updates(actor_params, updates)
    
    # Update target networks
    actor_target_params = soft_update(actor_params, actor_target_params)
    critic_target_params = soft_update(critic_params, critic_target_params)
    
    return actor_params, critic_params, actor_target_params, critic_target_params, actor_opt_state, critic_opt_state

# Simulation
@partial(jax.jit, static_argnums=(0,))
def simulate_episode(initial_state, actor, steps, rng):
    """
    Args:
        initial_state: shape (5,)
        steps: int
    Returns:
        list of (state, action, reward, next_state) tuples
    """
    state = initial_state
    trajectory = []
    for _ in range(steps):
        action = actor(state)  # Deterministic policy
        dW = jax.random.normal(rng, shape=state.shape[:1]) * jnp.sqrt(dt)
        next_state = step_state(state, action, dW)
        reward = compute_reward(state, action)
        trajectory.append((state, action, reward, next_state))
        state = next_state
    return trajectory

class DDPGAgent:
    def __init__(self, state_dim, action_dim, rng_key):
        # Initialize networks
        self.actor = Actor(action_dim=action_dim)
        self.critic = Critic()
        
        # Create network parameters
        rng_key, actor_key, critic_key = jax.random.split(rng_key, 3)
        
        # Initialize parameters with properly shaped dummy inputs
        dummy_state = jnp.ones([1, state_dim])  # Batch dimension of 1
        dummy_action = jnp.ones([1, action_dim])
        
        self.actor_params = self.actor.init(actor_key, dummy_state)
        self.critic_params = self.critic.init(critic_key, dummy_state, dummy_action)
        
        self.target_actor_params = self.actor_params
        self.target_critic_params = self.critic_params
        
        # Create optimizers
        self.actor_opt = build_actor(state_dim)
        self.critic_opt = build_critic(state_dim, action_dim)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

    @partial(jax.jit, static_argnums=(0,))
    def select_action(self, state, noise_scale=0.1, rng_key=None):
        """
        Args:
            state: shape (state_dim,)
            noise_scale: exploration noise scale
        Returns:
            action: shape (action_dim,)
        """
        # Ensure state has batch dimension
        state_batch = state.reshape(1, -1)
        action = self.actor.apply(self.actor_params, state_batch)
        if rng_key is not None:
            noise = jax.random.normal(rng_key, action.shape) * noise_scale
            action = jnp.clip(action + noise, -1.0, 1.0)
        return action[0]  # Remove batch dimension

def train_ddpg(
    agent,
    n_episodes=1000,
    steps_per_episode=100,
    batch_size=64,
    initial_state_fn=None,
    rng_key=jax.random.PRNGKey(0)
):
    """Train DDPG agent on multiple episodes"""
    buffer = ReplayBuffer(100000, 5, 1)
    
    @partial(jax.jit, static_argnums=(2,))
    def run_episode(rng_key, initial_state, steps):
        """Vectorized episode simulation"""
        def body_fn(carry, _):
            state, episode_rng = carry
            action_rng, noise_rng, step_rng, new_rng = jax.random.split(episode_rng, 4)
            
            # Select action with exploration noise
            action = agent.select_action(state, 0.1, action_rng)
            
            # Simulate one step
            dW = jax.random.normal(noise_rng) * jnp.sqrt(dt)
            next_state = step_state(state, action, dW)
            reward = compute_reward(state[None], action[None])[0]
            
            return (next_state, new_rng), (state, action, reward, next_state, jnp.array(False))
        
        episode_rng, init_rng = jax.random.split(rng_key)
        final_state, trajectory = jax.lax.scan(
            body_fn, (initial_state, init_rng), None, length=steps
        )
        # Unpack trajectory into separate arrays
        states, actions, rewards, next_states, dones = jax.tree_map(
            lambda x: jnp.array(x), list(zip(*trajectory))
        )
        return states, actions, rewards, next_states, dones
    
    # Training loop
    for episode in range(n_episodes):
        # Generate episode
        episode_key, train_key, rng_key = jax.random.split(rng_key, 3)
        initial_state = initial_state_fn(episode_key) if initial_state_fn else jnp.zeros(5)
        states, actions, rewards, next_states, dones = run_episode(
            episode_key, initial_state, steps_per_episode
        )
        
        # Store transitions in buffer
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            buffer.add(s, a, r, ns, d)
        
        # Training step if enough samples
        if len(buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            agent.actor_params, agent.critic_params, agent.target_actor_params, \
            agent.target_critic_params, agent.actor_opt_state, agent.critic_opt_state = train_step(
                states, actions, rewards, next_states, dones,
                agent.actor, agent.critic,
                agent.actor_params, agent.critic_params,
                agent.target_actor_params, agent.target_critic_params,
                agent.actor_opt_state, agent.critic_opt_state,
                agent.actor_opt, agent.critic_opt,
                train_key
            )

if __name__ == "__main__":
    agent = DDPGAgent(state_dim=5, action_dim=1, rng_key=jax.random.PRNGKey(0))
    train_ddpg(agent, n_episodes=1000, steps_per_episode=100)
    print("Training complete.")
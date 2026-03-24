from collections import defaultdict
from source.environment import ResidualEnv
from source.reupload_agent import generate_model_policy, qubits, observables
import tensorflow as tf
import numpy as np


# Initialize the Environment
env = ResidualEnv(num_assets=10)

# 2. Get the first state
state, _ = env.reset()

batch_size = 10
n_episodes = 1000
gamma = 0.99
env_kwargs = {'num_assets': 10}

optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
w_in, w_var, w_out = 0, 1, 2  # Make sure these match your model.trainable_variables indices!

model = generate_model_policy(qubits, n_layers = 3, n_bits = 20, rate=1.0, observables)


# 3. The RL Step Loop
for day in range(100):
    # Pass 40-feature state to Quantum Model (need to reshape for TF batch dim)
    # probabilities = quantum_model(state.reshape(1, 40))
    
    # Mocking the quantum output for now: generating 20 random bits
    action_bits = env.action_space.sample() 
    
    # 4. Pass the 20 bits back to the Environment
    next_state, reward, done, _, _ = env.step(action_bits)
    
    print(f"Day {day} | Reward: {reward:.4f} | Tiers Chosen: {env.current_tiers}")
    
    # (Here you would calculate the Loss and update the Quantum Gradients)
    
    state = next_state


def gather_episodes(model, batch_size, env_kwargs):
    trajectories = [defaultdict(list) for _ in range(batch_size)]
    # Instantiate your custom environment
    envs = [ResidualEnv(**env_kwargs) for _ in range(batch_size)]

    done = [False for _ in range(batch_size)]
    states = [env.reset()[0] for env in envs]

    while not all(done):
        unfinished_ids = [i for i in range(batch_size) if not done[i]]
        current_states = [states[i] for i in unfinished_ids]

        for i, state in zip(unfinished_ids, current_states):
            trajectories[i]['states'].append(state)

        # Compute 20 probabilities for all unfinished envs
        state_tensor = tf.convert_to_tensor(current_states, dtype=tf.float32)
        action_probs = model(state_tensor).numpy()

        states = [None for _ in range(batch_size)]
        for i, probs in zip(unfinished_ids, action_probs):
            # NEW: Sample 20 independent bits [1, 0, 1, 1...] based on probabilities
            action = np.random.binomial(n=1, p=probs)
            
            # Step the environment
            states[i], reward, terminated, truncated, _ = envs[i].step(action)
            done[i] = terminated or truncated
            
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    return trajectories


def compute_returns(rewards_history, gamma):
    """Compute discounted returns with discount factor `gamma`."""
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()

    return returns


@tf.function
def reinforce_update(states, actions, returns, model):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out],
                            [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

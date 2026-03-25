import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import multiprocessing
import tensorflow as tf

cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(cores)
tf.config.threading.set_intra_op_parallelism_threads(cores)
print(f"Forcing TensorFlow to use all {cores} CPU cores!")

from collections import defaultdict
from environment import ResidualEnv
from reupload_agent import generate_model_policy, qubits, observables, num_assets, n_bits, n_layers
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 10
n_episodes = 1000
gamma = 0.99
env_kwargs = {'num_assets': num_assets}

optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
w_in, w_var, w_out = 0, 1, 2

# Build the model
model = generate_model_policy(qubits, n_layers=n_layers, n_actions=n_bits, 
                              beta=1.0, observables=observables )


def gather_episodes(model, batch_size, env_kwargs, max_steps=30):
    trajectories = [defaultdict(list) for _ in range(batch_size)]
    envs = [ResidualEnv(**env_kwargs) for _ in range(batch_size)]

    done = [False for _ in range(batch_size)]
    states = [e.reset()[0] for e in envs]
    
    # NEW: Track how many steps each environment has taken
    step_counts = [0 for _ in range(batch_size)] 

    while not all(done):
        unfinished_ids = [i for i in range(batch_size) if not done[i]]
        current_states = [states[i] for i in unfinished_ids]

        for i, state in zip(unfinished_ids, current_states):
            trajectories[i]['states'].append(state)

        state_tensor = tf.convert_to_tensor(current_states, dtype=tf.float32)
        action_probs = model(state_tensor).numpy()

        states = [None for _ in range(batch_size)]
        for i, probs in zip(unfinished_ids, action_probs):
            action = np.random.binomial(n=1, p=probs)
            states[i], reward, terminated, truncated, _ = envs[i].step(action)
            
            # NEW: Force truncation if we hit the max step limit
            step_counts[i] += 1
            if step_counts[i] >= max_steps:
                truncated = True
                
            done[i] = terminated or truncated
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    return trajectories


def compute_returns(rewards_history, gamma):
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    return returns.tolist()


# REMOVED @tf.function (See point 3 below for why)
@tf.function(reduce_retracing=True)
def reinforce_update(states, actions, returns, model):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        probs = model(states)
        action_probs = actions * probs + (1.0 - actions) * (1.0 - probs)
        log_probs = tf.reduce_sum(tf.math.log(action_probs + 1e-8), axis=1)
        loss = -tf.reduce_mean(log_probs * returns)

    grads = tape.gradient(loss, model.trainable_variables)
    
    # NEW: Safely route gradients to the correct optimizer based on variable name
    for var, grad in zip(model.trainable_variables, grads):
        if 'lambdas' in var.name:
            optimizer_in.apply_gradients([(grad, var)])
        elif 'thetas' in var.name:
            optimizer_var.apply_gradients([(grad, var)])
        elif 'obs-weights' in var.name:
            optimizer_out.apply_gradients([(grad, var)])


# ============================================
# TRAINING LOOP
# ============================================
episode_reward_history = []

for batch in range(n_episodes // batch_size):
    trajectories = gather_episodes(model, batch_size, env_kwargs, max_steps=30)

    all_states = []
    all_actions = []
    all_returns = []

    for traj in trajectories:
        returns = compute_returns(traj['rewards'], gamma)
        all_states.extend(traj['states'])
        all_actions.extend(traj['actions'])
        all_returns.extend(returns)

    states_tensor = tf.convert_to_tensor(all_states, dtype=tf.float32)
    actions_tensor = tf.convert_to_tensor(all_actions, dtype=tf.float32)
    returns_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)

    reinforce_update(states_tensor, actions_tensor, returns_tensor, model)

    total_rewards = [sum(traj['rewards']) for traj in trajectories]
    avg_reward = np.mean(total_rewards)
    episode_reward_history.append(avg_reward)

    if (batch + 1) % 10 == 0:
        print(f"Batch {batch+1}/{n_episodes // batch_size} | "
              f"Avg Reward: {avg_reward:.4f}")

plt.plot(episode_reward_history)
plt.xlabel('Batch')
plt.ylabel('Average Reward')
plt.title('Training Progress')
plt.show()


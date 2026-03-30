import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import multiprocessing
import cirq
import sympy 
import numpy as np
import tensorflow as tf 
import tensorflow_quantum as tfq

# Force TensorFlow to use all available Kaggle CPU cores
cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(cores)
tf.config.threading.set_intra_op_parallelism_threads(cores)
print(f"Forcing TensorFlow to use all {cores} CPU cores!")

from variational_eigensolver import eigen_circuit

def portfolio_hamiltonian(qubits, mu_today, cov_matrix, risk_aversion=0.5):
    # ... [Keep your exact hamiltonian logic here] ...
    hamiltonian = cirq.PauliSum()
    num_assets = len(mu_today)
    weights = []

    for i in range(num_assets): 
        q_1 = qubits[2*i]
        q_2 = qubits[2*i + 1]

        x_1 = 0.5 - 0.5 * cirq.Z(q_1)
        x_2 = 0.5 - 0.5 * cirq.Z(q_2)

        w_i = 1.0 * x_1 + 2.0 * x_2
        weights.append(w_i)

        hamiltonian -= mu_today[i] * w_i

    for i in range(num_assets): 
        for j in range(num_assets):
            hamiltonian += risk_aversion * cov_matrix[i, j] * weights[i] * weights[j]
    
    return hamiltonian


def portfolio_optimisation(mu_today, var_today, recent_returns_df, ansatz_circuit, param_strings): 

    num_assets = len(mu_today)

    std_devs = np.sqrt(var_today)
    correlation_mat = recent_returns_df.corr().to_numpy()
    cov_matrix = np.outer(std_devs, std_devs) * correlation_mat

    qubits = sorted(ansatz_circuit.all_qubits())
    hamiltonian = portfolio_hamiltonian(qubits, mu_today, cov_matrix)

    circuit_tensor = tfq.convert_to_tensor([ansatz_circuit])
    expectation_layer = tfq.layers.Expectation(differentiator=tfq.differentiators.Adjoint())

    theta = tf.Variable(np.random.uniform(0, 2*np.pi, len(param_strings)), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    # ----------------------------------------------------
    # NEW: Wrapping the optimization step in tf.function
    # ----------------------------------------------------
    @tf.function(reduce_retracing=True)
    def train_step():
        with tf.GradientTape() as tape:
            energy = expectation_layer(
                circuit_tensor,
                symbol_names=param_strings,
                symbol_values=[theta],
                operators=hamiltonian
            )
            loss = energy[0][0]

        grads = tape.gradient(loss, [theta])
        optimizer.apply_gradients(zip(grads, [theta]))
        return loss

    # Execute the fast compiled graph
    for step in range(50):
        train_step()
    # ----------------------------------------------------
    
    resolver = cirq.ParamResolver(dict(zip(param_strings, theta.numpy())))
    resolved_circuit = cirq.resolve_parameters(ansatz_circuit, resolver)

    # Standard Cirq simulator is perfectly fine here since it only runs ONCE
    simulator = cirq.Simulator()
    result = simulator.simulate(resolved_circuit)

    state_probs = np.abs(result.state_vector())**2
    best_state_idx = np.argmax(state_probs)

    binary_string = format(best_state_idx, f'0{len(qubits)}b')
    bits = [int(b) for b in binary_string]

    optimal_tiers = []
    for i in range(num_assets):
        tier = 1 * bits[2*i] + 2*bits[2*i+1] 
        optimal_tiers.append(tier)

    return optimal_tiers

    
    




    
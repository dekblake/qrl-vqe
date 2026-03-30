import cirq
import numpy as np
import tensorflow as tf 
import tensorflow_quantum as tfq
from variational_eigensolver import eigen_circuit

# =================================================================
# 1. GLOBAL SETUP: Define these ONCE so they never re-compile
# =================================================================
expectation_layer = tfq.layers.Expectation(differentiator=tfq.differentiators.Adjoint())
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
global_theta = None # We will reuse this variable every day

@tf.function(reduce_retracing=True)
def fast_train_step(circuit_tensor, hamiltonian_tensor, theta, param_strings):
    with tf.GradientTape() as tape:
        energy = expectation_layer(
            circuit_tensor,
            symbol_names=param_strings,
            symbol_values=[theta],
            operators=hamiltonian_tensor
        )
        loss = energy[0][0]
        
    grads = tape.gradient(loss, [theta])
    optimizer.apply_gradients(zip(grads, [theta]))
    return loss


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
    global global_theta # Pull in the global variable

    num_assets = len(mu_today)
    std_devs = np.sqrt(var_today)
    correlation_mat = recent_returns_df.corr().to_numpy()
    cov_matrix = np.outer(std_devs, std_devs) * correlation_mat

    qubits = sorted(ansatz_circuit.all_qubits())
    hamiltonian = portfolio_hamiltonian(qubits, mu_today, cov_matrix)

    circuit_tensor = tfq.convert_to_tensor([ansatz_circuit])
    
    # 2. Convert Hamiltonian to a 2D Tensor so tf.function accepts it without retracing
    hamiltonian_tensor = tf.expand_dims(tfq.convert_to_tensor([hamiltonian]), axis=0)

    # 3. Reuse the same theta variable every day, just give it new random starting weights
    # 3. Reuse the same theta variable every day, but randomize the starting weights
    if global_theta is None:
        global_theta = tf.Variable(np.random.uniform(0, 2*np.pi, len(param_strings)), dtype=tf.float32)
    else:
        global_theta.assign(np.random.uniform(0, 2*np.pi, len(param_strings)).astype(np.float32))

    # --> NEW: Wipe Adam's momentum from yesterday so today is a fresh estimate
    if len(optimizer.variables()) > 0:
        for var in optimizer.variables():
            var.assign(tf.zeros_like(var))

    # 4. BLAST THROUGH 50 STEPS USING THE GLOBALLY COMPILED GRAPH
    for step in range(50):
        fast_train_step(circuit_tensor, hamiltonian_tensor, global_theta, param_strings)
    
    # Get final parameters
    resolver = cirq.ParamResolver(dict(zip(param_strings, global_theta.numpy())))
    resolved_circuit = cirq.resolve_parameters(ansatz_circuit, resolver)

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




    
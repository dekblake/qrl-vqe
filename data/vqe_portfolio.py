import circuit
import sympy 
import numpy as np
import tensorflow as tf 
import tensorflow_quantum as tfq

from source.variational_eigensolver import eigen_circuit

def portfolio_hamiltonian(qubits, mu_today, cov_matrix, risk_aversion=0.5):

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

        hamiltonian -= mu_todaya[i] * w_i

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
    




    
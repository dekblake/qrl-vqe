import cirq 
import sympy 
from math import pi 
import numpy as np

from cirq.contrib.svg import SVGCircuit


def eigen_circuit(qubits, layer_count, seed):
  #Generates the circuit for the eigensolver, RA-pairwise

  rng = np.random.default_rng(seed)

  qubit_count = len(qubits)
  
  params = sympy.symbols(f'x(0:{layer_count})' + f'_(0:{qubit_count})')
  params = np.asarray(params).reshape((layer_count, qubit_count))

  circuit = cirq.Circuit()

  #initial sqrt of Hadamard gate
  circuit += [cirq.H(q) for q in qubits]

  for l in range(layer_count):

    #Uniform random distribution of Rx,Ry,Rz gates + weights
    pauli_gate = rng.choice([cirq.rx, cirq.ry, cirq.rz], qubit_count)
    circuit += [pauli_gate[i](params[l, i])(q) for i,q in enumerate(qubits)]

    #pauli-two entanglement topology (pairwise-TwoLocal)
    circuit += [cirq.CZ(q0, q1) for q0,q1 in zip(qubits, qubits[1:])]

  return circuit 

"""qubit_count, layer_count = 5, 3
qubits = cirq.GridQubit.rect(1, qubit_count)
circuit = eigen_circuit(qubits, layer_count, 1)
SVGCircuit(circuit)""
import tensorflow as tf 
import tensorflow_quantum as tfq
import cirq
import numpy as np 
import sympy
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

#tf.get_logger().setLevel('ERROR')

def Rotation(qubit, symbols):
    #gates that rotatate classical data points around bloch sphere
    
    return [
        cirq.rx(symbols[0])(qubit),    
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit)
    ]

def Entangling(qubits):
    #Circular entanglement topology (CZ gates)

    circular_cz = [cirq.CZ(q0, q0) for q0, q1 in zip(qubits, qubits[1:])]
    circular_cz += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])

    return circular_cz

def VQCircuit(qubits, n_layers): 
    #The data-reuploading PQC

    number_qubits = len(qubits)

    #symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*number_qubits})')
    params = np.asarray(params).reshape ((n_layers +1 , number_qubits, 3))

    #symbols for encoding angles: four features
    x_inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{number_qubits})')
    x_inputs = np.asarray(x_inputs).reshape((n_layers, number_qubits, ))
    y_inputs = sympy.symbols(f'y(0:{n_layers})' + f'_(0:{number_qubits})')
    y_inputs = np.asarray(y_inputs).reshape((n_layers, number_qubits))
    z_inputs = sympy.symbols(f'z(0:{n_layers})' + f'_(0:{number_qubits})')
    z_inputs = np.asarray(z_inputs).reshape((n_layers, number_qubits))

    #literal circuit: bloch variational layer, encoding, free variational layer
    circuit = cirq.Circuit()
    for l in range(n_layers): 

        #variational layer
        circuit += cirq.Circuit(
                    Rotation(q, params[l,i]) for i,q in enumerate(qubits)
                    )
        circuit += Entangling(qubits)

        #encoding layer
        circuit += cirq.Circuit(cirq.rx(x_inputs[l, i])(q) for i,q in enumerate(qubits))
        circuit += cirq.Circuit(cirq.ry(y_inputs[l, i])(q) for i,q in enumerate(qubits))
        circuit += cirq.Circuit(cirq.ry(z_inputs[l, i])(q) for i,q in enumerate(qubits))

        #last variational layer
        circuit += cirq.Circuit(Rotation(q, params[n_layers, i]) 
                                for i, q in enumerate(qubits))

def ControlledParameterisedQCircuit(qubits, n_layers):

    return
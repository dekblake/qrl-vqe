import tensorflow as tf
import tensorflow_quantum as tfq
import gymnasium as gym
import cirq
import sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

tf.get_logger().setLevel('ERROR')

def Rotation(qubit, symbols):
    #gates that rotate classical data points around bloch sphere
    
    return [
        cirq.rx(symbols[0])(qubit),    
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit)
    ]

def Entangling(qubits):
    #Circular entanglement topology (CZ gates)

    circular_cz = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    circular_cz += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])

    return circular_cz

def VQCircuit(qubits, n_layers): 
    #The data-reuploading PQC

    number_qubits = len(qubits)

    #symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*number_qubits})')
    params = np.asarray(params).reshape((n_layers + 1 , number_qubits, 3))

    #symbols for encoding angles: four features
    x_inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{number_qubits})')
    x_inputs = np.asarray(x_inputs).reshape((n_layers, number_qubits, ))
    y_inputs = sympy.symbols(f'y(0:{n_layers})' + f'_(0:{number_qubits})')
    y_inputs = np.asarray(y_inputs).reshape((n_layers, number_qubits))
    z1_inputs = sympy.symbols(f'z(0:{n_layers})' + f'_A(0:{number_qubits})')
    z1_inputs = np.asarray(z1_inputs).reshape((n_layers, number_qubits))
    z2_inputs = sympy.symbols(f'z(0:{n_layers})' + f'_B(0:{number_qubits})')
    z2_inputs = np.asarray(z2_inputs).reshape((n_layers, number_qubits))

    #literal circuit: bloch variational layer, encoding, free variational layer
    circuit = cirq.Circuit()
    for l in range(n_layers): 

        #initial variational layer
        circuit += cirq.Circuit(
                    Rotation(q, params[l,i]) for i,q in enumerate(qubits)
                    )
        #circular entanglement topology
        circuit += Entangling(qubits)

        #encoding layer + data inputs
        circuit += cirq.Circuit(cirq.rx(x_inputs[l, i])(q) for i,q in enumerate(qubits))
        circuit += cirq.Circuit(cirq.ry(y_inputs[l, i])(q) for i,q in enumerate(qubits))
        circuit += cirq.Circuit(cirq.rz(z1_inputs[l, int(i/2)])(q) for i,q in enumerate(qubits) if i % 2 == 0)
        circuit += cirq.Circuit(cirq.rz(z2_inputs[l, int((i-1)/2)])(q) for i,q in enumerate(qubits) if i % 2 == 1)

        #last variational layer + weights
        circuit += cirq.Circuit(Rotation(q, params[n_layers, i]) 
                                for i, q in enumerate(qubits))
      #flatten inputs
    args = (x_inputs, y_inputs, z1_inputs, z2_inputs)
    inputs = np.concatenate(args).ravel().tolist()   

    return circuit, list(params.ravel().tolist()), list(inputs)

"""
n_qubits, n_layers = 4, 1
qubits = cirq.GridQubit.rect(1, n_qubits)
circuit, _, _, = VQCircuit(qubits, n_layers)
SVGCircuit(circuit)
"""

#nothing changed
class VQCReuploading(tf.keras.layers.Layer):
    #Applies variational angles and scaling parameters

    def __init__(self, 
                 qubits,
                 n_layers,
                 observables,
                 activation='linear',
                 name="re-uploading_PQC"):
        super(VQCReuploading, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = VQCircuit(qubits, n_layers)
        
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(initial_value=self.theta_init(
            shape=(1, len(theta_symbols)), dtyoe='float32'),
                                        trainable=True,
                                        name='thetas')
        lmbd_init = tf.ones(shape=(self.n_qubits*self.n_layers,))
        self.lmbd = tf.Variable(initial_value=lmbd_init,
                                dtype='float32',
                                trainable=True,
                                name='lambdas')

        # symbol order
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs): 
        #inputs[0] = encoding data for the state

        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim,1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum('i, ji->', self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(
            self.activation)(scaled_inputs)
        
        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])
    

#Softmax Policy Gradient
class Nonalternating(tf.keras.layers.Layer): 

    #defining observable weights
    def __init__(self, output_dim): 
        super(Nonalternating, self).__init__()
        self.w = tf.Variable(initial_value=tf.constant,
                             dtype='float32',
                             trainable=True,
                             name="obs-weights")
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w)


n_qubits = 4  # Dimension of the state vectors in CartPole
n_layers = 3  # Number of layers in the PQC
n_bits = 20  # Number of actions in CartPole

qubits = cirq.GridQubit.rect(1, n_qubits)    

observables = [cirq.Z(q) for q in qubits]

def generate_model_policy(qubits, n_layers, n_actions, beta, observables):
    #keras model for data reuploading policy

    input_tensor = tf.keras.Input(shape=(len(qubits),),
                                  dtype=tf.dtypes.float32,
                                  name='input')
    re_uploading_pqc = VQCReuploading(qubits, n_layers, 
                                      observables)([input_tensor])
    process = tf.keras.Sequential([
        Nonalternating(n_actions),
        tf.keras.layers.Lambda(lambda x: x*beta),
        tf.keras.layers.Softmax()
    ],
        name='observable-policy')
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model

model = generate_model_policy(qubits, n_layers, n_bits, 1.0, observables)


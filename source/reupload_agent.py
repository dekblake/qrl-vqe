import tensorflow as tf 
import numpy as np 
import tensorflow_quantum as tfq
import cirq
import sympy

from source.reupload_pqc import VQCircuit


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
        self.theta = tf.Variable(initial_value=theta_init(
            shape=(1, len(theta_symbols)), dtype='float32'),
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
        scaled_inputs = tf.einsum('i, ji->ji', self.lmbd, tiled_up_inputs) #changed
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
        self.w = tf.Variable(initial_value=tf.ones(shape=(1, output_dim), dtype='float32'), #initial method changed
                             dtype='float32',
                             trainable=True,
                             name="obs-weights")
    
    def call(self, inputs):
        return inputs * self.w #to avoid mixing of 20 qubits before scaling


n_qubits = 20  # Dimension of the state vectors in CartPole
n_layers = 3  # Number of layers in the PQC
n_bits = 20  # Number of actions in CartPole

qubits = cirq.GridQubit.rect(1, n_qubits)    

observables = [cirq.Z(q) for q in qubits]

def generate_model_policy(qubits, n_layers, n_actions, beta, observables):
    #keras model for data reuploading policy

    #increased inputs for selected market features 
    input_tensor = tf.keras.Input(shape=(40,),
                                  dtype=tf.dtypes.float32,
                                  name='input')
    re_uploading_pqc = VQCReuploading(qubits, n_layers, 
                                      observables)([input_tensor])
    process = tf.keras.Sequential([
        Nonalternating(n_actions),
        tf.keras.layers.Lambda(lambda x: x*beta),
        tf.keras.layers.Activation('Sigmoid')  #changed for combinatorial QUBO
    ],
        name='observable-policy')
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model

model = generate_model_policy(qubits, n_layers, n_bits, 1.0, observables)
import numpy as np

class Layer():
    '''
    Layer neuron, very simple and ez, shouldnt actually instantiated!
    '''

    def __init__(self, n):
        '''
        n: number of neurons in the layer
        '''
        if n <= 0:
            raise ValueError('inputs must have length > 0')

        self.n = n
        self.type = 'Generic'

    def __str__(self):
        neuron_string = f'{self.type} Layer\n--------\n'
        return neuron_string

class InputLayer(Layer):
    '''
    Input layer, simple, no need for weights or biases
    '''
    def __init__(self, n):
        super(InputLayer, self).__init__(n)
        self.type = 'Input'

    def __str__(self):
        '''
        Input Layer
        --------
        (_) (_) (_) (_) (_) (_) (_) (_)
        '''
        neuron_string = super().__str__()
        neuron_string += ' '.join(['(_)' for _ in range(self.n)])

        return neuron_string

    def fire(self, inputs: np.ndarray):
        '''
        fire the neurons into the input layer
        '''
        # verify inputs is one d array
        if len(inputs.shape) != 1:
            raise ValueError(f'inputs must be a one dimensional array instead of {len(inputs.shape)} dimensions')
        n = len(inputs)
        if n != n:
            raise ValueError(f'inputs must have same length as {n}')

        self.activations = np.array(inputs)
        return self.activations

class OutputLayer0():
    '''
    Output Layer of one neuron (good for simple linear regression), simple, needs weights and one bias
    Technically not even a layer because its just one output
    '''

    def __init__(self,
                 weights: np.ndarray,
                 bias: float | np.number,
                 activation_function = 'none'):
        self.type = 'Output0'

        # verify that data types are corect
        if len(weights.shape) != 1:
            raise ValueError(f'weights must be a one dimensional array')

        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.activation = np.nan

    def __str__(self):
        '''
        Output0 Layer
        --------
        weights: b0, w0, w1, w2
        (_)
        '''
        neuron_string = f'{self.type} Layer\n--------\n'
        neuron_string += f'weights: {self.bias}, {', '.join(map(str, self.weights))}'
        neuron_string += '\n(_)\n'

        return neuron_string

    def fire(self, prev_activations: np.ndarray):
        if len(prev_activations) != len(self.weights):
            raise ValueError('previous layer and weights must be of same length')
        self.activation = sum((self.weights * prev_activations)) + self.bias
        return self.activation

class DenseLayer(Layer):
    '''
    basic dense layer that can contain 1 to inf neurons

    self.weights: np.ndarray
    self.biases: np.ndarray
    self.n: int | np.number
    '''

    def __init__(self,
                 n: int | np.number,
                 weights: np.ndarray,
                 biases: np.ndarray,
                 activation_function: str='none'):
        '''
        n = size of dnese layer
        weights = weight matrix
                            ex. ((w, w, w),
                                 (w, w, w))
        '''

        super(DenseLayer, self).__init__(n)
        self.type = 'Dense'
        if not (len(weights.shape) == 2 and weights.shape[0] == n and len(biases.shape) == 1 and n == len(biases)):
            raise ValueError(f'weigts or biases shape does not alight with n -> f{n}')

        self.n = n
        self.weights = weights
        self.biases = biases

    def fire(self, prev_activations: np.ndarray):
        '''
        fire the neuron in the dense layer
        '''

        self.weights: np.ndarray
        self.biases: np.ndarray
        self.n: int | np.number

        if len(prev_activations.shape) != 1 or prev_activations.shape[0] != self.weights.shape[1]:
            raise ValueError(f'prev_activations dimensions {prev_activations.shape}, doesnt work with weights shape {self.weights.shape}')

        self.activations = self.weights.dot(prev_activations) + self.biases
        return self.activations

class OutputLayer(DenseLayer):
    '''
    Output layer basically same as dense layer
    need for weights and biases
    '''
    def __init__(self,
                 n: int | np.number,
                 weights: np.ndarray,
                 biases: np.ndarray,
                 activation_function: str='none'):
        super(OutputLayer, self).__init__(n=n, weights=weights, biases=biases, activation_function=activation_function)
        self.type = 'OutputLayer'
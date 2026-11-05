from network import InputLayer, OutputLayer0, OutputLayer, DenseLayer
import numpy as np
import numpy.random as random

def test_standard_in3_out2():
    '''
    first test:
    test working firing with input layer of 3 and output layer of 2

    features [age, height, times_arrested]
    labels [wage, favorite number]
    '''
    input_layer = InputLayer(3)
    output_layer = OutputLayer(n=2,
                               weights=np.array(
                                   [
                                       [1.1, .5, -.1],
                                       [10, -30, 28]
                                   ]
                               ),
                               biases=np.array(
                                   [2, 50]
                               ))
    
    activations = input_layer.fire(np.array([19, 6, 1]))
    assert (activations == np.array([19, 6, 1])).all()

    activations = output_layer.fire(activations)
    assert (activations == np.array([25.8, 88])).all()

def test_standard_in30_out_50():
    '''
    second standard test:
    test working firing with input layer of 30 and output layer of 50

    features and labels both random
    '''
    weights = random.random((50, 30))
    print(weights.shape)
    biases = random.random(50)
    print(biases.shape)

    input_layer = InputLayer(30)
    output_layer = OutputLayer(n=50,
                               weights=weights,
                               biases=biases)
    
    inputs = random.random(30)
    activations = input_layer.fire(inputs)
    assert (activations == inputs).all()

    outputs = weights.dot(inputs) + biases

    activations = output_layer.fire(activations)
    assert (activations == outputs).all()

def test_dense_in30_out_50():
    '''
    first dense test:
    test working firing with input layer of 30 and output layer of 50

    features and labels both random
    '''
    weights = random.random((50, 30))
    print(weights.shape)
    biases = random.random(50)
    print(biases.shape)

    input_layer = InputLayer(30)
    dense_layer = DenseLayer(n=50,
                               weights=weights,
                               biases=biases)
    
    inputs = random.random(30)
    activations = input_layer.fire(inputs)
    assert (activations == inputs).all()

    outputs = weights.dot(inputs) + biases

    activations = dense_layer.fire(activations)
    assert (activations == outputs).all()

def test_basic():
    '''
    test InputLayer and OutputLayer0 only
    basic tests
    '''
    my_layer = InputLayer(2)
    output_layer = OutputLayer0(
        weights=np.array([1_900.0, 333]),
        bias=2
    )
    print(my_layer, '\n')
    print(output_layer)

    activations = my_layer.fire(np.array([20, 6]))
    print(activations)

    assert (activations == np.array([20, 6])).all()

    activations = output_layer.fire(activations)
    print(activations)

    assert activations == 40_000

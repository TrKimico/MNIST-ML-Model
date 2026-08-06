import numpy as np

def matrix_multiplication(neuron_width,input_width,dataset,neuron_layer,weight_layer):
    """The mathematical operations to process the matrix multiplication of input by weight"""
    W = weight_layer.reshape(neuron_width, input_width)
    neuron_layer += dataset @ W.T
    return neuron_layer

def matrix_addition(neuron_layer,bias_layer):
    """The mathematical operations to process the matrix addition of [input*weight] and bias"""
    neuron_layer += bias_layer
    return neuron_layer

def ReLu(neuron_layer):
    """The mathematical operation to trigger neural activation of hidden layers"""
    neuron_layer[neuron_layer < 0] = 0
    return neuron_layer

def SoftMax(neuron_layer):
    """Transforms the last neuron layer into a probability spread"""
    row_max = np.max(neuron_layer, axis=1, keepdims=True)   # shape (batch, 1)
    shifted = neuron_layer - row_max                        # normalize for stability
    exp_vals = np.exp(shifted)
    output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    return output            

def hidden_layer_computation(neuron_width,input_width,dataset,neuron_layer,weight_layer, bias_layer):
    """Concatenates all the necessary steps to obtain the output of a hidden layer for a batch size"""
    output = matrix_multiplication(neuron_width,input_width,dataset,neuron_layer,weight_layer)
    output = matrix_addition(output,bias_layer)
    output = ReLu(output)
    return output

def output_layer_computation(neuron_width,input_width,dataset,neuron_layer,weight_layer, bias_layer):
    """Concatenates all the necessary steps to obtain the output of the LAST layer for a batch size"""
    output = matrix_multiplication(neuron_width,input_width,dataset,neuron_layer,weight_layer)
    output = matrix_addition(output,bias_layer)
    output = SoftMax(output)
    return output

##################################################
# Metrics
##################################################

def loss_function(batch, dataset, expected_value, settings, weight0, weight1, weight2):
    """Uses multiclass cross-entropy functions to find how far from expected output are the final outputs"""
    correct_probs = dataset[np.arange(batch), expected_value]
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)
    cross_entropy = -np.mean(np.log(correct_probs))

    L2_term = (settings.l2_lambda / 2) * (np.sum(weight0.layer**2) + np.sum(weight1.layer**2) + np.sum(weight2.layer**2))

    loss = cross_entropy + L2_term
    return loss

def compute_accuracy(neuron, expected_value):
    """Counts how many predictions in the batch match the expected class"""
    predicted = np.argmax(neuron.layer, axis=1)
    correct = np.sum(predicted == expected_value)
    return correct
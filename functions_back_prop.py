import numpy as np

def error_signal_hidden(previous_error_signal,previous_error_signal_width,error_signal,hidden_layer,hidden_layer_width,hidden_layer_weight):
    """Computes the error signal for each neuron of a hidden layer"""
    W = hidden_layer_weight.reshape(previous_error_signal_width, hidden_layer_width)
    constant = previous_error_signal @ W  # shape (batch, hidden_layer_width)
    # multiply by derivative of ReLU: 1 where activation != 0, else 0
    error_signal = constant * (hidden_layer != 0)
    return error_signal

def error_signal_output(batch,error_signal,error_signal_width,output_layer,expected_value):
    """Computes the initial error signal for each neuron of the output layer"""
    one_hot = np.zeros((batch, error_signal_width))
    one_hot[np.arange(batch), expected_value] = 1
    error_signal = output_layer - one_hot
    return error_signal

def compute_weight_cost(settings, error_signal,weight_cost,previous_layer,weight):
    """Computes the weight cost for each neuron of the output layer"""
    # error_signal.T @ previous_layer gives shape (destination_width, origin_width),
    # flattened C-order as neuron_idx*origin_width + input_idx — matching w.layer's forward-prop layout
    weight_cost = (error_signal.T @ previous_layer) / settings.batch
    weight_cost = weight_cost.flatten() + settings.l2_lambda * weight.layer # L2 penalty to counter overfitting
    return weight_cost.flatten()

def compute_bias_cost(batch,error_signal):
    """Average the bias costs across the batch"""
    bias_cost = error_signal.sum(axis=0) / batch
    return bias_cost

def update_parameter(settings,parameter,cost,t):
    """Apply the final parameter cost to the initial parameters with Adam Algorithm"""
    # update the attributes value for the parameter
    parameter.moment = settings.beta1 * parameter.moment + (1 - settings.beta1) * cost 
    parameter.variance = settings.beta2 * parameter.variance + (1 - settings.beta2) * cost**2
    # bias correct
    m_hat = parameter.moment / (1 - settings.beta1**t)
    v_hat = parameter.variance / (1 - settings.beta2**t)
    # compute the final value
    parameter.layer -= settings.learning_rate * m_hat / ((v_hat)**0.5 + settings.epsilon)
    return parameter
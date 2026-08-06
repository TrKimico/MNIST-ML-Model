import math
import random
from random import gauss
import numpy as np
from .settings import Settings

settings = Settings()
# fix the seed for reproductibility
random.seed(settings.seed)
np.random.seed(settings.seed)

# STRUCTURAL OBJECTS
class Neuron():
    """A Class that defines how neurons look and behave"""
    def __init__(self,settings,layer_length):
        """A Method that defines the core caracteristics of a neuron"""
        self.layer_length = layer_length
        self.layer        = self.layer = np.zeros((settings.batch, self.layer_length))
        self.current_batch_size = settings.batch

    def reset(self, batch_size):
        """Handles gracefully the batch size of the last batch of the epoch which may be smaller"""
        self.layer        = self.layer = np.zeros((batch_size, self.layer_length))
        self.current_batch_size = batch_size

class Weight():
    """A Class that defines how weights look and behave"""
    def __init__(self,origin_layer_length,destination_layer_length):
        """A Method that defines the core caracteristics of a weight"""
        self.origin_layer_length      = origin_layer_length
        self.destination_layer_length = destination_layer_length
        # uses the He algorithm for initialisation : optimised for ReLU activation
        # python initialisation isn't ideal but it proves more performant than np on this instance
        self.layer    = np.array([gauss(0, math.sqrt(2/self.origin_layer_length)) 
                        for _ in range(self.origin_layer_length * self.destination_layer_length)])
        self.moment   = np.zeros(self.origin_layer_length * self.destination_layer_length) # Adam parameter
        self.variance = np.zeros(self.origin_layer_length * self.destination_layer_length) # Adam parameter

class Bias():
    """A Class that defines how biases look and behave"""
    def __init__(self,layer_length):
        self.layer_length = layer_length
        self.layer        = np.zeros(layer_length)
        self.moment       = np.zeros(layer_length) # Adam parameter
        self.variance     = np.zeros(layer_length) # Adam parameter

# BACKPROPAGATION MATHEMATICAL OBJECTS
class Cost_Weight():
    """A Class that defines how Weight Costs look and behave"""
    def __init__(self,origin_layer_length,destination_layer_length):
        """A Method that defines the core characteristics of a Weight Cost"""
        self.origin_layer_length      = origin_layer_length
        self.destination_layer_length = destination_layer_length
        self.layer                    = np.zeros(origin_layer_length*destination_layer_length)

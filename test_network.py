# import libraries
from sklearn.datasets import fetch_openml
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
import math
# import other modules
import functions_forward_prop as fp
from classes.settings import Settings
from classes.network_objects import Neuron

np.set_printoptions(legacy='1.25') # otherwise it doesn't print floats correctly in the terminal
sample_size = 15

##################################################
# Import and Format Data
##################################################

# IMPORT AND HANDLE THE DATASET
## import
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float32)
Y = mnist.target.astype(np.int64)
## keep only the test dataset
X_TEST, Y_TEST = X[60000:]/255, Y[60000:]

# import the weights and biases
data = np.load("best_params.npz")
# format it
w0_layer = data['w0']
w1_layer = data['w1']
w2_layer = data['w2']
b0_layer = data['b0']
b1_layer = data['b1']
b2_layer = data['b2']

##################################################
# Initialise Network Objects
##################################################

settings = Settings()
# NEURONE (reference the whole layer by nx.layer)
n0 = Neuron(settings,settings.layer0_width)
n1 = Neuron(settings,settings.layer1_width)
n2 = Neuron(settings,settings.output_width)

##################################################
# Run a sample through the network
##################################################

# shuffle dataset and pick only a handful for test and display
indices = np.arange(X_TEST.shape[0])
np.random.shuffle(indices)
X_TEST = X_TEST[indices]
Y_TEST = Y_TEST[indices]
X_TEST = X_TEST[:sample_size]
Y_TEST = Y_TEST[:sample_size]

# reset batch length
batch = len(X_TEST) 
n0.reset(batch)
n1.reset(batch)
n2.reset(batch)

# Forward Propagation
n0.layer = fp.hidden_layer_computation(settings.layer0_width,settings.input_width,X_TEST,n0.layer,w0_layer,b0_layer)
n1.layer = fp.hidden_layer_computation(settings.layer1_width,settings.layer0_width,n0.layer,n1.layer,w1_layer,b1_layer)
n2.layer = fp.output_layer_computation(settings.output_width,settings.layer1_width,n1.layer,n2.layer,w2_layer,b2_layer)

answers = [np.argmax(n2.layer[i]) for i in range(len(X_TEST))]

##################################################
# Plot Results
##################################################

n_cols = 5
n_rows = math.ceil(sample_size / n_cols)

plt.figure(figsize=(12, 3 * n_rows))
for i in range(sample_size):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(X_TEST[i].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {answers[i]}, Answer:{Y_TEST[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
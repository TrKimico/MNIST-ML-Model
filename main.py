# External Libraries
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import copy
# Access other Modules
import functions_forward_prop as fp
import functions_back_prop as bp
## file structure is important
from classes.settings import Settings
from classes.network_objects import Neuron, Weight, Bias, Cost_Weight

##################################################
# Import and Format data
##################################################

np.set_printoptions(legacy='1.25') # otherwise it doesn't print floats correctly in the terminal
## import
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float32)
Y = mnist.target.astype(np.int64)
## format
X_TRAIN, X_VAL = X[:55000]/255, X[55000:60000]/255
Y_TRAIN, Y_VAL = Y[:55000], Y[55000:60000]
X_TEST, Y_TEST = X[60000:]/255, Y[60000:]

##################################################
# Initialize the Network Objects
##################################################

settings = Settings()

# fix the seed for reproductibility 
np.random.seed(settings.seed)

# NEURONE (reference the whole layer by nx.layer)
n0 = Neuron(settings,settings.layer0_width)
n1 = Neuron(settings,settings.layer1_width)
n2 = Neuron(settings,settings.output_width)
# WEIGHT (reference the whole layer by wx.layer)
w0 = Weight(settings.input_width,settings.layer0_width)
w1 = Weight(settings.layer0_width,settings.layer1_width)
w2 = Weight(settings.layer1_width,settings.output_width)
# BIAS (reference the whole layer by bx.layer)
b0 = Bias(settings.layer0_width)
b1 = Bias(settings.layer1_width)
b2 = Bias(settings.output_width)
# COST WEIGHT
cw0 = Cost_Weight(settings.input_width,settings.layer0_width)
cw1 = Cost_Weight(settings.layer0_width,settings.layer1_width)
cw2 = Cost_Weight(settings.layer1_width,settings.output_width)
# COST BIAS (computed from error)
# ERROR : Neurons that hold the error signal instead of the activation signal(reference the whole layer by ex.layer)
e0 = Neuron(settings,settings.layer0_width)
e1 = Neuron(settings,settings.layer1_width)
e2 = Neuron(settings,settings.output_width)

##################################################
# Define Important Functions
##################################################

def shuffle_dataset(training_set, answers):
    """Changes the indice of all the dataset items for each Epoch"""
    indices = np.arange(training_set.shape[0])
    np.random.shuffle(indices)
    training_set = training_set[indices]
    answers = answers[indices]
    return training_set, answers

def train_batch(TEST,RESULT,w0,w1,w2,b0,b1,b2,timestep_counter):
    """Trains the network over one batch of data, returns the updated parameters"""
    # Initialise
    timestep_counter += 1 # Counter for Adam
    batch = len(TEST) 
    n0.reset(batch)
    n1.reset(batch)
    n2.reset(batch)
    e0.reset(batch)
    e1.reset(batch)
    e2.reset(batch)
    # Forward Propagation
    n0.layer = fp.hidden_layer_computation(settings.layer0_width,settings.input_width,TEST,n0.layer,w0.layer,b0.layer)
    n1.layer = fp.hidden_layer_computation(settings.layer1_width,settings.layer0_width,n0.layer,n1.layer,w1.layer,b1.layer)
    n2.layer = fp.output_layer_computation(settings.output_width,settings.layer1_width,n1.layer,n2.layer,w2.layer,b2.layer)
    # Gradual Descent
    ## error signal computation (= bias cost computation)
    e2.layer = bp.error_signal_output(batch,e2.layer,settings.output_width,n2.layer,RESULT)
    e1.layer = bp.error_signal_hidden(e2.layer,settings.output_width,e1.layer,n1.layer,settings.layer1_width,w2.layer)
    e0.layer = bp.error_signal_hidden(e1.layer,settings.layer1_width,e0.layer,n0.layer,settings.layer0_width,w1.layer)
    ## weight cost computation
    cw2.layer = bp.compute_weight_cost(settings,e2.layer,cw2.layer,n1.layer,w2) #batch,e2.layer,cw2.layer,n1.layer
    cw1.layer = bp.compute_weight_cost(settings,e1.layer,cw1.layer,n0.layer,w1)
    cw0.layer = bp.compute_weight_cost(settings,e0.layer,cw0.layer,TEST,w0)
    ## bias cost computation
    cb2 = bp.compute_bias_cost(batch,e2.layer)
    cb1 = bp.compute_bias_cost(batch,e1.layer)
    cb0 = bp.compute_bias_cost(batch,e0.layer)
    # Update Parameters
    w0 = bp.update_parameter(settings,w0,cw0.layer,timestep_counter)
    w1 = bp.update_parameter(settings,w1,cw1.layer,timestep_counter)
    w2 = bp.update_parameter(settings,w2,cw2.layer,timestep_counter)
    b0 = bp.update_parameter(settings,b0,cb0,timestep_counter)
    b1 = bp.update_parameter(settings,b1,cb1,timestep_counter)
    b2 = bp.update_parameter(settings,b2,cb2,timestep_counter)
    return w0,w1,w2,b0,b1,b2,timestep_counter

def train_network(DATASET,ANSWERS,w0,w1,w2,b0,b1,b2,timestep_counter):
    """Trains the Entire Dataset once, updating parameters at every batch"""
    DATASET, ANSWERS = shuffle_dataset(DATASET, ANSWERS)
    index_position = 0
    while index_position < len(DATASET):
        if index_position + settings.batch > len(DATASET): # handles the last batch that would be indexed out of range
            batch_X = DATASET[index_position:]
            batch_Y = ANSWERS[index_position:]
        else:
            batch_X = DATASET[index_position:index_position+settings.batch]
            batch_Y = ANSWERS[index_position:index_position+settings.batch]
        w0,w1,w2,b0,b1,b2,timestep_counter = train_batch(batch_X, batch_Y, w0,w1,w2,b0,b1,b2,timestep_counter)
        index_position += settings.batch
    return w0,w1,w2,b0,b1,b2,timestep_counter

def test_network(DATASET,ANSWERS,w0,w1,w2,b0,b1,b2):
    """Runs the network forward on the testing set to test for accuracy after each Epoch"""
    loss_list = []
    accuracy = 0
    index_position = 0
    while index_position < len(DATASET):
        if index_position + settings.batch > len(DATASET): # handles the last batch that would be indexed out of range
            batch_X = DATASET[index_position:]
            batch_Y = ANSWERS[index_position:]
        else:
            batch_X = DATASET[index_position:index_position+settings.batch]
            batch_Y = ANSWERS[index_position:index_position+settings.batch]
        # Initialisation
        batch = len(batch_X)
        n0.reset(batch)  # always reset, regardless of size match
        n1.reset(batch)
        n2.reset(batch)
        n0.layer = fp.hidden_layer_computation(settings.layer0_width,settings.input_width,batch_X,n0.layer,w0.layer,b0.layer)
        n1.layer = fp.hidden_layer_computation(settings.layer1_width,settings.layer0_width,n0.layer,n1.layer,w1.layer,b1.layer)
        n2.layer = fp.output_layer_computation(settings.output_width,settings.layer1_width,n1.layer,n2.layer,w2.layer,b2.layer)
        # Compute Accuracy
        accuracy += fp.compute_accuracy(n2,batch_Y) # only the slice of the batch
        batch_loss = fp.loss_function(batch,n2.layer,batch_Y,settings,w0,w1,w2)
        loss_list.append(batch_loss)
        index_position += settings.batch
    final_loss = sum(loss_list) / len(loss_list)
    accuracy /= len(DATASET)
    return accuracy,final_loss

##################################################
# Execute The Program
##################################################

######### initialise #########

best_val_accuracy = 0
best_params = None
accuracy_val_list    = []
accuracy_train_list  = []
loss_val_list        = []
loss_train_list      = []
timestep_counter     = 0      # needed for Adam
benchmark_accuracy   = 0.9780 # compared to the average output of the pytorch_benchmark module

######### test before any training #########

accuracy,loss = test_network(X_TEST,Y_TEST,w0,w1,w2,b0,b1,b2)
print(f"The accuracy of the untrained model is : {round(accuracy*100,4)}%")
print(f"The loss of the untrained model is : {round(loss,4)}\n")
# run another test on training data to add as baseline for data plot
baseline_accuracy_val, baseline_loss_val = test_network(X_VAL,Y_VAL,w0,w1,w2,b0,b1,b2)
loss_val_list.append(baseline_loss_val)
accuracy_val_list.append(baseline_accuracy_val)
baseline_accuracy_train, baseline_loss_train = test_network(X_TRAIN,Y_TRAIN,w0,w1,w2,b0,b1,b2)
loss_train_list.append(baseline_loss_train)
accuracy_train_list.append(baseline_accuracy_train)

#########  run the entire training dataset through the network then test it #########

for x in range(settings.epoch_length):
    w0,w1,w2,b0,b1,b2,timestep_counter = train_network(X_TRAIN,Y_TRAIN,w0,w1,w2,b0,b1,b2,timestep_counter)
    accuracy_train,loss_train = test_network(X_TRAIN,Y_TRAIN,w0,w1,w2,b0,b1,b2)
    accuracy_train_list.append(accuracy_train)
    loss_train_list.append(loss_train)
    accuracy_val,loss_val = test_network(X_VAL,Y_VAL,w0,w1,w2,b0,b1,b2)
    print(f"Epoch({x+1}) accuracy is : {round(accuracy_val*100,4)}%")
    loss_val_list.append(loss_val)
    accuracy_val_list.append(accuracy_val)

    # checkpoint if this is the best so far
    if accuracy_val > best_val_accuracy:
        best_val_accuracy = accuracy_val
        best_params = copy.deepcopy((w0,w1,w2,b0,b1,b2))
        print(f"New best val accuracy : {round(accuracy_val*100,4)}% — checkpointed\n")
    # Exit if the estimated best value for the model is reached
    if len(accuracy_val_list) >= 5 and len(set(accuracy_val_list[-5:])) == 1:
        print("\nMaximum plateau has been reached, exiting training")
        break

    # Learning Rate Decay : attemps to push past high accuracy plateau
    if x > 0 and x % 2 == 0:
        settings.learning_rate = settings.initial_learning_rate * settings.decay_rate**(x/settings.decay_steps)

######### test after training #########

w0,w1,w2,b0,b1,b2 = best_params # restore best value stored during training

accuracy,loss = test_network(X_TEST,Y_TEST,w0,w1,w2,b0,b1,b2)
print(f"\nThe accuracy of the model is : {round(accuracy*100,4)}%")
print(f"\nThe loss of the model is : {round(loss,4)}\n")

if accuracy > benchmark_accuracy:
    print(f"The model outperformed the benchmark by a margin of {round((accuracy-benchmark_accuracy)*100,4)}%")
elif (benchmark_accuracy*1.001) < accuracy < (benchmark_accuracy*0.999):
    print("The model performed very similarily to the benchmark")
else:
    print(f"The model was outperformed by the benchmark by a margin of {round((benchmark_accuracy-accuracy)*100,4)}%")

while True:
    export_paramz = input("\nDo you wish to export the weights and biases? (Y/N): ").strip().upper()
    if export_paramz in {"Y", "YES"}:
        print("Exporting Parameters\n")
        np.savez("best_params.npz", w0=w0.layer, w1=w1.layer, w2=w2.layer, b0=b0.layer, b1=b1.layer, b2=b2.layer)
    elif export_paramz in {"N", "NO"}:
        print("Skipping Exportation\n")
    else:
        print("Please enter a valid input.")
        continue
    break

######### plot data #########

##### plot the accuracy of TRAIN and VAL
# settings for the display of the graph
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.figure(figsize=(15, 11))
plt.xlabel("Epoch",fontsize=20, color='k')
plt.ylabel("Accuracy from training data",fontsize=20, color='k')
plt.title("Model Accuracy over Epoch",fontsize=25, color='r')
indices_val       = list(range(len(accuracy_val_list)))
# plot data
## VAL accuracy
x = indices_val
y = accuracy_val_list
plt.ylim(0.9,1.0)
plt.plot(x, y, color='k', marker='.', label='VAL accuracy')
## TRAIN accuracy
i = indices_val
j = accuracy_train_list
plt.ylim(0.9,1.0)
plt.plot(i, j, color='r', marker='.', label='TRAIN accuracy')
plt.legend(fontsize=14)
# print it on screen
plt.show()

##### plot the loss of TRAIN and VAL
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.figure(figsize=(15, 11))
plt.xlabel("Epoch",fontsize=20, color='k')
plt.ylabel("Loss from training data",fontsize=20, color='k')
plt.title("Model Loss over Epoch",fontsize=25, color='r')
indices_val       = list(range(len(accuracy_val_list)))
# plot data
## VAL loss
x = indices_val
y = loss_val_list
plt.plot(x, y, color='k', marker='.', label='VAL Loss')
plt.yscale('log')
## TRAIN loss
i = indices_val
j = loss_train_list
plt.plot(i, j, color='r', marker='.', label='TRAIN Loss')
plt.yscale('log')
plt.legend(fontsize=14)
# print it on screen
plt.show()
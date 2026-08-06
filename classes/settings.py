class Settings():
    def __init__(self):
        """the structural variables of the settings"""
        # Network Structure
        self.depth         = 2
        self.input_width   = 28 * 28 # all the pixels in each image
        self.layer0_width  = 128     # n0
        self.layer1_width  = 128     # n1
        self.output_width  = 10      # n2

        # Training Structure
        self.batch         = 128
        self.epoch_length  = 50
        self.seed          = 40

        # Learning Variables
        self.initial_learning_rate = 0.01
        self.learning_rate         = 0.01  # updated each epoch
        self.decay_rate            = 0.4
        self.decay_steps           = 2   # how long it takes for learning_rate to decrease by a factor of decay_rate
        self.l2_lambda             = 5e-4  # governs L2 penalty

        # Adam variables
        self.beta1         = 0.9
        self.beta2         = 0.999
        self.epsilon       = 1e-8 

# CALCULATIONS FOR GRADUAL DESCENT

# WEIGHTS
# δC/δw(input) = a_i(input) × ReLU'(z_j(n0)) × Σ_k[ w_(jk)(n1) × error_k(n1) ]
# δC/δw(n0) = a_i(n0) × ReLU'(z_j(n1)) × Σ_k[ w_(jk)(n2) × error_k(n2) ]
# δC/δw_(n1) = a_i(n1) × (ŷ_j − y_j)

# BIASES (Relu' * ERROR except for n2)
# δC/δb(n0) = ReLU'(z_j(n0)) * Σ_k[ w_(jk)(n1) × error_k(n1) ]
# δC/δb(n1) = ReLU'(z_j(n1)) * Σ_k[ w_(jk)(n2) × error_k(n2) ]
# δC/δb(n2) = (ŷ_j − y_j)

# ERROR
# δC/δa(n0) = Σ_k[ w_(jk)(n1) × error_k(n1) ]
# δC/δa(n1) = Σ_k[ w_(jk)(n2) × error_k(n2) ]
# δC/δa(n2) = (ŷ_j − y_j)
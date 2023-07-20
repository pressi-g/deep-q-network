# Author: Preston Govender

"""
Hyperparameters for the Deep Q Network
"""

### MODEL HYPERPARAMETERS 
numActions = 3               # 3 possible actions: left, right, move forward
inputSize = 49               # size of the flattened input state (7x7 matrix of tile IDs)

### TRAINING HYPERPARAMETERS
alpha = 0.0002               # learning_rate
episodes = 5000              # Total episodes for training
batch_size = 128             # Neural network batch size
target_update = 20000        # Number of episodes between updating target network

# Q learning hyperparameters
gamma = 0.90                 # Discounting rate

# Exploration parameters for epsilon greedy strategy
start_epsilon = 1.0          # exploration probability at start
stop_epsilon = 0.01          # minimum exploration probability 
decay_rate = 20000           # exponential decay rate for exploration prob

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size # Number of experiences stored in the Memory when initialized for the first time
memorySize = 500000          # Number of experiences the Memory can keep - 500000

### TESTING HYPERPARAMETERS
# Evaluation hyperparameter
evalEpisodes = 1000          # Number of episodes to be used for evaluation

# Change this to 'False' if you only want to evaluate a previously trained agent
train = True                 # Specify True to train a model, otherwise only evaluate
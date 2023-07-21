import yaml

# Load the config from the YAML file
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Access hyperparameters from the config dictionary
numActions = config["numActions"]
inputSize = config["inputSize"]
alpha = config["alpha"]
episodes = config["episodes"]
batch_size = config["batch_size"]
target_update = config["target_update"]
gamma = config["gamma"]
start_epsilon = config["start_epsilon"]
stop_epsilon = config["stop_epsilon"]
decay_rate = config["decay_rate"]
pretrain_length = config["pretrain_length"]
memorySize = config["memorySize"]
evalEpisodes = config["evalEpisodes"]
train = config["train"]

# Start the training loop if train is True, otherwise start the evaluation loop
if train:
    # Training code here
else:
    # Evaluation code here
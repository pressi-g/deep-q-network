import yaml
from deep_q_learning import deep_q_learning

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

def main():

    # Start the training loop if train is True, otherwise start the evaluation loop
    if train:
        # Training code here
        # probably something like: 
        pass
        # deep_q_learning(episodes, alpha, gamma, start_epsilon, stop_epsilon, decay_rate, pretrain_length, memorySize, batch_size, target_update, inputSize, numActions)
    else:
        # Evaluation code here
        # probably something like:
        # evaluate():
        pass

if __name__ == "__main__":
    main()


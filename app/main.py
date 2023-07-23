from train import train_model
from evaluate import evaluate_model

# set random seed
import random

random.seed(5)

# Set the variable to indicate whether to train or evaluate: False for evaluate, True for train
train = False  # or False

if __name__ == "__main__":
    if train:
        train_model()
    else:
        evaluate_model()

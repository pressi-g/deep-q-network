# Deep Q-Network (DQN) on MiniGridWorld

This repository contains an implementation of Deep Q-Network (DQN) applied to the MiniGridWorld problem using the Farama-Foundation's [minigrid](https://github.com/Farama-Foundation/Minigrid) Gymnasium environment. The agent is restricted to three actions: forward, turn left, and turn right.

The MiniGridWorld environment consists of a 2D grid representing an empty room. The agent's goal is to navigate the grid and reach the green goal square, which provides a sparse reward. The agent receives a small penalty for the number of steps taken to reach the goal.

<div style="display: flex; justify-content: center;">
  <img src="app/dqn.gif" alt="solution">
</div>

## Introduction

Deep Q-Network (DQN) is a reinforcement learning algorithm that combines the Q-learning algorithm with deep neural networks. It overcomes the limitations of tabular Q-learning by approximating the action-value function using a neural network. DQN has been successful in solving complex tasks by learning directly from high-dimensional input, such as images.

In this implementation, we use the MiniGridWorld environment as the training and evaluation environment. The agent learns by interacting with the environment, observing the state, and taking actions to maximize its cumulative reward.

The DQN algorithm uses an experience replay buffer to store and sample past experiences, enabling more efficient learning and reducing correlations between consecutive samples. It also employs a target network to stabilize learning by using a separate network for estimating the target Q-values.

## Installation

To run this code, you need to have Python 3.8 and pip installed on your system. Here are the steps to set up the environment:

1. Create a virtual environment using `conda` or `venv`:

   ```shell
   conda create -n rl-env python=3.8
   ```

   or

   ```shell
   python3 -m venv rl-env
   ```

   Replace `rl-env` with any name you prefer for your virtual environment. `conda` is recommended since this project was developed using `conda`.

2. Clone this repository:

   ```
   git clone https://github.com/pressi-g/deep-q-network
   ```

3. Change into the project directory:

   ```
   cd deep-q-network
   ```

4. Activate the virtual environment:

   ```shell
   conda activate rl-env
   ```

   or

   ```shell
   source rl-env/bin/activate
   ```

5. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

6. To view and render the optimal policy run the `render_optimal_policy.py` script:

   ```
   python3 render_optimal_policy.py
   ```

   This will load the trained model and render the optimal policy of the agent in the MiniGridWorld environment.

7. To evaluate the trained agent, run the `main.py` script with the `train` variable set to `False`:

   ```
   python3 main.py
   ```

   This will load the trained model and evaluate the agent's performance in the MiniGridWorld environment.

8. To train agent from scratch, run main.py but change the `train` variable to `True`:

   ```
   python3 main.py
   ```

   This will train the agent using the default hyperparameters and save the trained model.


You can modify the hyperparameters in the `train.py` script to experiment with different settings and observe their impact on the agent's learning.

## Directory Structure

The directory structure of this repository is as follows:

```
└── app
|   └── models
|   |   └── trained_model.pth           # Directory to save the trained model file
|   ├── __init__.py
|   ├── deep_q_learning.py              # Script for the DQN implementation
|   ├── evaluate.py                     # Script for evaluating the trained DQN agent
|   ├── main.py                         # Main script to control training or evaluation
|   ├── render_optimal_policy.py        # Script to visualize the optimal policy of DQN agent
|   ├── train.py                        # Script for training the DQN agent
|   └── utils.py                        # Utility functions used in the implementation
└── requirements
|     └── requirements.txt              # File containing required Python packages
|     └── test_requirements.txt         # File containing required packages for testing
└── .gitignore                          # File specifying files/directories to ignore in version control
└── README.md                           # Readme file with information about the project
└── tests
   └── unit-tests                       # Directory containing unit test scripts
   |   ├── ...
   |   └── ...
   └── integration-tests                # Directory containing integration test scripts
      ├── ...
      └── ...

```

Here's a brief description of the files and directories in this repository:

- `app`: Contains the main application code.
  - `models`: Directory to save the trained model file (`trained_model.pth`).
  - `__init__.py`
  - `deep_q_learning.py`: Script for the DQN implementation.
  - `evaluate.py`: Script for evaluating the trained DQN agent.
  - `main.py`: Main script to control training or evaluation.
  - `render_optimal_policy.py`: Script to visualize the optimal policy of the DQN agent.
  - `train.py`: Script for training the DQN agent.
  - `utils.py`: Utility functions used in the implementation.
- `requirements`: Contains required Python dependencies for the project.
  - `requirements.txt`: File containing required Python packages.
  - `test_requirements.txt`: File containing required packages for testing.
- `.gitignore`: File specifying files/directories to ignore in version control.
- `README.md`: Readme file with information about the project.
- `tests`: Contains unit and integration tests for the application.
  - `unit-tests`: Directory containing unit test scripts.
  - `integration-tests`: Directory containing integration test scripts.

Feel free to explore the code files, modify them, and experiment with different settings to understand and improve the Deep Q-Network (DQN) algorithm on MiniGridWorld.

## Testing

This project includes test cases to ensure the correctness of the implemented algorithms and functionalities. The tests are written using the `pytest` framework. To run the tests, follow these steps:

1. Make sure you have installed the project dependencies as mentioned in the installation section.

2. Open a terminal or command prompt and navigate to the project directory.

3. Install `pytest` using `pip`:

   ```shell
   pip install pytest
   ```

4. Once `pytest` is installed, you can run the tests by executing the following command:

   ```shell
   pytest
   ```

   This command will discover and execute all the test cases in the project.

   Note: The test files should be named with the prefix `test_` for `pytest` to automatically detect them.

5. After running the tests, you will see the test results in the terminal or command prompt, indicating which tests passed or failed.

If you encounter any issues or failures during the test execution, please feel free to open an issue in this repository for assistance.
# # standard imports
# import numpy as np
# import random
# from os.path import exists

# # RL imports
# import gymnasium as gym
# import minigrid
# from minigrid.wrappers import *
# from collections import namedtuple, deque

# # torch imports
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# # tensorboard imports
# from torch.utils.tensorboard import SummaryWriter

# # import functions
# from utils import *


# ### MODEL HYPERPARAMETERS
# numActions = 3               # 3 possible actions: left, right, move forward
# inputSize = 49               # size of the flattened input state (7x7 matrix of tile IDs)

# ### TRAINING HYPERPARAMETERS
# alpha = 0.001               # learning_rate
# episodes = 3000              # Total episodes for training
# batch_size = 64             # Neural network batch size
# target_update = 10000        # Number of episodes between updating target network

# # Q learning hyperparameters
# gamma = 0.9                 # Discounting rate

# # Exploration parameters for epsilon greedy strategy
# start_epsilon = 0.9         # exploration probability at start
# stop_epsilon = 0.05          # minimum exploration probability
# decay_rate = 3000           # exponential decay rate for exploration prob

# ### MEMORY HYPERPARAMETERS
# pretrain_length = batch_size # Number of experiences stored in the Memory when initialized for the first time
# memorySize = 100000          # Number of experiences the Memory can keep - 500000

# ### TESTING HYPERPARAMETERS
# # Evaluation hyperparameter
# evalEpisodes = 1000          # Number of episodes to be used for evaluation

# # Change this to 'False' if you only want to evaluate a previously trained agent
# train = False

# # Update the number of episodes if training on GPU
# episodes = device_specific_episodes(episodes)

# # Set the random seed for reproducible experiments
# random.seed(5)

# # define some functions
# ## Function to e-greedily select next action based on current state
# def select_action(state):
#     # generate a random number
#     sample = random.random()

#     # calculate the epsilon threshold, based on the epsilon-start value, the epsilon-stop value,
#     # the number of training steps taken and the epsilon decay rate
#     # here we are using an exponential decay rate for the epsilon value
#     eps_threshold = stop_epsilon+(start_epsilon-stop_epsilon)*math.exp(-1. * steps_done / decay_rate)

#     # compare the generated random number to the epsilon threshold
#     if sample > eps_threshold:
#         # act greedily towards the Q-values of our policy network, given the state

#         # we do not want to gather gradients as we are only generating experience, not training the network
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(state).max(1)[1].unsqueeze(0)
#     else:
#         # select a random action with equal probability
#         return torch.tensor([[random.randrange(numActions)]], device=device, dtype=torch.long)

# def optimize_model():

#     # Perform one step of the optimization
#     if len(memory) < batch_size:
#         return

#     # Sample a mini-batch from the replay memory
#     transitions = memory.sample(batch_size)
#     batch = Transition(*zip(*transitions))

#     # Prepare the data for optimization
#     state_batch = torch.cat(batch.currentState).to(device)
#     action_batch = torch.cat(batch.action).to(device)

#     # Compute Q-values for the current state-action pairs
#     state_action_values = policy_net(state_batch).gather(
#         1, action_batch)


#     reward_batch = torch.tensor(batch.reward) #, device=device, dtype=torch.float).unsqueeze(1)
#     non_final_next_states = torch.cat([s for s in batch.nextState if s is not None])
#     # Compute the expected Q-values (TD-targets) for the next states
#     next_state_values = torch.zeros(batch_size, device=device)
#     non_final_mask = torch.tensor(
#         tuple(map(lambda s: s is not None, batch.nextState)),
#         device=device, dtype=torch.bool)
#     next_state_values[non_final_mask] = target_net(
#         non_final_next_states).max(1)[0].detach()

#     TDtargets = (next_state_values * gamma) + reward_batch

#     # Compute the TD-errors and the loss
#     TDerrors = TDtargets.unsqueeze(1) - state_action_values

#     # Configure the MSELoss
#     criterion = nn.MSELoss()

#     # Calculate the loss for the mini-batch.
#     # Make sure to resize the TD-targets to be the same shape as the state-action values tensor
#     loss = criterion(state_action_values, TDtargets.unsqueeze(1))

#     # zero the gradients
#     optimizer.zero_grad()

#     # calculate the gradients by backpropagation from the loss function
#     loss.backward()

#     # We clamp the gradients in the policy network to the range [-1,1]
#     # This also helps keep training stable
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)

#     # Update the network parameters of the policy network using the optimizer
#     optimizer.step()
# # Make the environment
# env = create_minigrid_environment()

# # use wrapper to only extract observation
# env = ImgObsWrapper(env)

# # Instantiate the policy network and the target network
# hiddenLayerSize = (128,128)
# policy_net = DQN(inputSize, numActions, hiddenLayerSize)
# target_net = DQN(inputSize, numActions, hiddenLayerSize)

# # # Copy the weights of the policy network to the target network
# # target_net.load_state_dict(policy_net.state_dict())

# # We don't want to update the parameters of the target network so we set it to evaluation mode
# target_net.eval()

# Transition = namedtuple('Transition',
#                         ('currentState', 'action', 'nextState', 'reward'))


# # Instantiate memory
# memory = ReplayMemory(memorySize)

# # Tensorboard writer
# writer = SummaryWriter()

# # Compute MSE loss
# criterion = nn.MSELoss()

# # Create the optimizer
# optimizer = optim.Adam(policy_net.parameters(), lr=alpha)

# global steps_done
# steps_done = 3

# max_steps = env.max_steps

# if train:
#     # Training loop
#     print('Start training...')
#     for episode in range(episodes):
#         obs, info = env.reset()

#         state = preprocess(obs)
#         total_reward = 0

#         for step in range(max_steps):
#             # Choose an action using epsilon-greedy strategy
#             # action = select_action_e_greedy(state, stop_epsilon, start_epsilon, decay_rate, steps_done, numActions, policy_net)
#             action = select_action(state)

#             # Take the chosen action in the environment
#             next_obs, reward, done, truncated, _ = env.step(action.item())

#             # Preprocess the next state
#             next_state = preprocess(next_obs)

#             if (done or truncated):
#                 next_state = None

#             # Store the transition in the replay memory
#             memory.push(state, action, next_state, torch.tensor([reward], device=device))

#             # Move to the next state
#             state = next_state
#             total_reward += reward

#             optimize_model()

#             # Update the target network
#             if steps_done % target_update == 0:
#                 update_target_net(policy_net, target_net)

#             # Episode finished when done or truncated is true
#             if done or truncated:
#                 # Record the reward and total training steps taken
#                 if done:
#                     print(f'Finished episode {episode + 1} successfully taking {step + 1} steps and receiving reward {total_reward}')
#                 else:
#                     print(f'Truncated episode {episode + 1} taking {step + 1} steps and receiving reward {total_reward}')
#                 break

#             # Update the number of steps taken
#             steps_done += 1

#         # Record the total reward for the episode
#         writer.add_scalar('Reward/Episode', total_reward, episode)
#         # Record the number of steps taken for the episode
#         writer.add_scalar('Steps/Episode', step + 1, episode)

#         # Save the policy network
#         # if episode % 100 == 0:

#     torch.save(policy_net.state_dict(), f'./models/{episode}.pth')

#     print('Finished training!')
#     writer.flush()
#     writer.close()

# else:
#     # Evaluation loop
#     print('Start evaluation...')
#     # Set the path to the saved model
#     model_path = 'models/2999.pth'

#     # Load the model
#     policy_net = DQN(inputSize, numActions, hiddenLayerSize)
#     load_model(policy_net, model_path)

#     # Initialize counters for evaluation metrics
#     finishCounter = 0.0
#     totalSteps = 0.0
#     totalReward = 0.0

#     # Run the evaluation loop
#     for e in range(evalEpisodes):
#         # Initialize the environment and state
#         currentObs, _ = env.reset()
#         currentState = preprocess(currentObs)

#         # the main RL loop
#         for i in range(0, env.max_steps):
#             # Select and perform an action
#             action = select_action(currentState)
#             a = action.item()

#             # Take action 'a', receive reward 'reward', and observe the next state 'obs'
#             # 'done' indicates if the termination state was reached
#             obs, reward, done, truncated, info = env.step(a)

#             if done or truncated:
#                 # Observe the new state
#                 nextState = None
#             else:
#                 nextState = preprocess(obs)

#             if done or truncated:
#                 totalReward += reward
#                 totalSteps += env.step_count
#                 if done:
#                     print('Finished evaluation episode %d with reward %f, %d steps, reaching the goal' % (e, reward, env.step_count))
#                     finishCounter += 1
#                 if truncated:
#                     print('Failed evaluation episode %d with reward %f, %d steps' % (e, reward, env.step_count))
#                 break

#             # Move to the next state
#             currentState = nextState

#     # Print a summary of the evaluation results
#     print('Completion rate %.2f with an average reward %0.4f and average steps %0.2f' % (finishCounter / evalEpisodes, totalReward / evalEpisodes, totalSteps / evalEpisodes))
#     print('Finished evaluation!')

# env.close()

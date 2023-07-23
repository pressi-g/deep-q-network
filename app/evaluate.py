from utils import *
from deep_q_learning import *

# Set the path to the saved model
model_path = 'models/2999.pth'

# Load the model
policy_net = DQN(inputSize, numActions, hiddenLayerSize)
load_model(policy_net, model_path)

# Initialize counters for evaluation metrics
finishCounter = 0.0
totalSteps = 0.0
totalReward = 0.0

# Run the evaluation loop
for e in range(evalEpisodes):
    # Initialize the environment and state
    currentObs, _ = env.reset()
    currentState = preprocess(currentObs)

    # the main RL loop
    for i in range(0, env.max_steps):
        # Select and perform an action
        action = select_action(currentState)
        a = action.item()

        # Take action 'a', receive reward 'reward', and observe the next state 'obs'
        # 'done' indicates if the termination state was reached
        obs, reward, done, truncated, info = env.step(a)

        if done or truncated:
            # Observe the new state
            nextState = None
        else:
            nextState = preprocess(obs)

        if done or truncated:
            totalReward += reward
            totalSteps += env.step_count
            if done:
                print('Finished evaluation episode %d with reward %f, %d steps, reaching the goal' % (e, reward, env.step_count))
                finishCounter += 1
            if truncated:
                print('Failed evaluation episode %d with reward %f, %d steps' % (e, reward, env.step_count))
            break

        # Move to the next state
        currentState = nextState

# Print a summary of the evaluation results
print('Completion rate %.2f with an average reward %0.4f and average steps %0.2f' % (finishCounter / evalEpisodes, totalReward / evalEpisodes, totalSteps / evalEpisodes))

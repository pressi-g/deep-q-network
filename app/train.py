

# Make the gym environment
env = gym.make('MiniGrid-Empty-8x8-v0')

# Use a wrapper so the observation only contains the grid information
env = ImgObsWrapper(env)

episodes = 2               # total number of training episodes
max_steps = env.max_steps  # maximum number of steps allowed before truncating episode
steps_done = 0             # total training steps taken

print('Start training...')
for e in range(episodes):
    
    # reset the environment
    obs, _ = env.reset()

    # extract the current state from the observation
    state = preprocess(obs)
    
    for i in range(0, max_steps):

        # Choose an action
        # Pick a random action
        action = select_action(state)
        a = action.item()
        
        # take action 'a', receive reward 'reward', and observe next state 'obs'
        # 'done' indicate if the termination state was reached
        obs, reward, done, truncated, info = env.step(a)
   
        # extract the next state from the observation
        nextState = preprocess(obs)
        
        # if the episode is finished, the nextState is set to None to indicate that the
        # <s,a,r,s'> transition led to a terminating state
        if (done or truncated):
            nextState = None
        
        # Store the transition <s,a,r,s'> in the replay memory

        # Move to the next state          
        currentState = nextState

        # Perform one step of the optimization (on the policy network) by
        # sample a mini-batch and train the model using the sampled mini-batch
        
        
        # If the target update threshold is reached, update the target network, 
        # copying all weights and biases in the policy network   

        
        # Episode finished when done or truncated is true
        if (done or truncated):
            # Record the reward and total training steps taken
            if (done):
                # if agent reached its goal successfully
                print('Finished episode successfully taking %d steps and receiving reward %f' % (env.step_count, reward))
            else:
                # agent failed to reach its goal successfully 
                print('Truncated episode taking %d steps and receiving reward %f' % (env.step_count, reward))
            break
            
        
print('Done training...')
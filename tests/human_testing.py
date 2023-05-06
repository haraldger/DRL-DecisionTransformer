import gym

# create the MsPacman-v0 environment
env = gym.make('ALE/MsPacman-v5', render_mode='human')

# initialize the environment
obs = env.reset()

# display the game screen
env.render()

# play the game manually using keyboard inputs
done = False
while not done:
    action = input("Enter action (0-8): ")  # get input from keyboard
    if action not in ["0", "1", "2", "3", "4", "5", "6", "7", "8"]:
        action = 0
    else:
        action = int(action)
    obs, reward, done, info, _ = env.step(action)  # take a step in the environment
    env.render()  # display the updated game screen

# close the environment
env.close()
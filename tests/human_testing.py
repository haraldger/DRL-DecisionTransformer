import gym
import matplotlib as plt
import cv2 as cv

# create the MsPacman-v0 environment
env = gym.make('ALE/MsPacman-v5', render_mode='human')

# initialize the environment
obs = env.reset()

# display the game screen
env.render()

inactive_frams = 65
for _ in range(inactive_frams):
    state, reward, done, info, _ = env.step(0)

# display the game screen
new_state = state[:-38,:,:]
print(new_state.shape)
cv.imshow("image v1", new_state)
new_state = cv.resize(new_state, (84,84), interpolation=cv.INTER_CUBIC)
print(new_state.shape)
cv.imshow("image v2", new_state)
cv.waitKey()

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
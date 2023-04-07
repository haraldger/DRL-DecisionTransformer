import gym
import time

env = gym.make('ALE/MsPacman-v5')
observation, _ = env.reset()
inactive_frames = 65

for _ in range(inactive_frames):
    action = 0  # noop
    env.step(action)

for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    print(observation.shape)

env.close()
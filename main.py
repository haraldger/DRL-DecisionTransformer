import sys
import time
import gym
from utils import experience_replay

env = gym.make('ALE/MsPacman-v5')
observation, _ = env.reset()
inactive_frames = 65

memory_size = 40000
memory = experience_replay.ReplayBuffer(capacity=memory_size, dims=observation.shape)

for _ in range(inactive_frames):
    action = 0  # noop
    env.step(action)

for i in range(10000):
    prev_observation = observation
    action = env.action_space.sample()

    # Execute new action
    observation, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    # Insert into replay memory
    memory.add(prev_observation, action, observation, reward)

    if i % 1000 == 0:
        print("Iteration: {}".format(i))

# Sample from replay memory
state_sample, action_sample, next_state_sample, reward_sample = memory.sample_tensor_batch(32)

memory.show()

print("State, action, next state, reward sample shapes:")
print(state_sample.shape)
print(action_sample.shape)
print(next_state_sample.shape)
print(reward_sample.shape)

print("State memory, action memory, next state memory, reward memory shapes:")
print(memory.state_memory.shape)
print(memory.action_memory.shape)
print(memory.next_state_memory.shape)
print(memory.reward_memory.shape)

env.close()
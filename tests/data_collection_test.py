import sys
import time
import gym
import numpy as np
from utils import experience_replay, data_collection, data_read


def run():
    print("Unit test for data_collection.py")
    print()
    data_collection.run_tests()
    print()

    print("Unit test for data_read.py")
    print()
    data_read.run_tests()
    print()

    print("Integration test for data_collection.py and data_read.py")
    print()

    env = gym.make('ALE/MsPacman-v5')
    observation, _ = env.reset()
    inactive_frames = 65

    data_collector = data_collection.DataCollector("test_main.h5")
    data_collector.set_init_state(observation)

    for _ in range(inactive_frames):
        action = 0  # noop
        env.step(action)

    for i in range(10000):
        prev_observation = observation
        action = env.action_space.sample()

        # Execute new action
        observation, reward, terminated, truncated, _ = env.step(action)
        data_collector.store_next_step(action, reward, observation, terminated)

        if terminated or truncated:
            observation, info = env.reset()
            data_collector.set_init_state(observation)
            print("done, iteration: ", i)




    data_reader = data_read.DataReader("test_main.h5")
    length = len(data_reader)
    print("Length: ", length)
    idx = 100
    # print("Index: ", idx)
    # state_sample, action_sample, reward_sample, next_state_sample, done_sample = data_reader[idx]

    env.close()

    print()
import sys
import time
import gym
import numpy as np
from utils import experience_replay, data_collection, data_read

def run_test():
        env = gym.make('ALE/MsPacman-v5')
        observation, _ = env.reset()
        inactive_frames = 65

        data_collector = data_collection.DataCollector("data_collecton_test_file.h5")
        data_collector.set_init_state(observation)

        expected_length = 0

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
                if terminated:
                    expected_length = i+1           
                observation, info = env.reset()
                data_collector.set_init_state(observation)
                print("done, iteration: ", i)


        data_reader = data_read.DataReader("data_collecton_test_file.h5")
        length = len(data_reader)
        print("Length: ", length)
        assert(length == expected_length)
        idx = 100
        print("Index: ", idx)
        state_sample, action_sample, reward_sample, next_state_sample, done_sample = data_reader[idx]


        env.close()
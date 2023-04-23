"""
Purpose: Works in tandum with data_collection.py
         Reads traj data in from a file 
"""
import sys
import numpy as np
import random
import h5py
import torch
import cv2 as cv

state_shape = (210, 160, 3)

class TrajectoryData:
    inital_state = None

    # Going to store sequence of (state, action, reward, next_state, done)
    data_pairs = []

    def __init__(self, init_state):
        self.inital_state = init_state
        self.data_pairs = []

    def add_iteration(self, action, reward, next_state, done):
        if len(self.data_pairs) == 0:
            self.data_pairs.append((self.inital_state, action, reward, next_state, done))
        else:
            self.data_pairs.append((self.data_pairs[-1][3], action, reward, next_state, done))


class DataReader:
    read_filename = None    
    all_traj_data = []

    def __init__(self, read_file):
        self.read_filename = read_file
        self.all_traj_data = []
    
    def process(self):
        with h5py.File(self.read_filename) as file:
            for ep in file.keys():
                episode = file[ep]
                
                # Read data from episode
                read_states_compressed = episode["states"][:]
                read_states = []
                for i in range(read_states_compressed.shape[0]):
                    read_states.append(cv.imdecode(np.frombuffer(read_states_compressed[i], dtype=np.uint8), cv.IMREAD_UNCHANGED))

                read_actions = episode["actions"][()]
                read_rewards = episode["rewards"][()]
                read_done = episode["done"][()]

                if len(read_actions) == 0:
                    continue

                # Process data into trajectory pairs
                traj_data = TrajectoryData(read_states[0])
                for i in range(0, len(read_actions)):
                    traj_data.add_iteration(read_actions[i], read_rewards[i], read_states[i+1], read_done[i])
                
                # Add epsiode trajectory pairs to all pairs
                self.all_traj_data += traj_data.data_pairs


if __name__ == "__main__":
    # Test with file from data_collection.py
    TEST_OUTPUT_FILENAME = "test_traj_long.h5"

    reader = DataReader(TEST_OUTPUT_FILENAME)
    reader.process()

    print("Number of data tuples: ", len(reader.all_traj_data))

    print("First action: ", reader.all_traj_data[0][1])
    print("First reward: ", reader.all_traj_data[0][2])
    print("First done value: ", reader.all_traj_data[0][4])
    print("Shape of state: ", reader.all_traj_data[0][0].shape)
    print("Shape of next state: ", reader.all_traj_data[0][3].shape)

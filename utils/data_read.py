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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

state_shape = (210, 160, 3)

class TrajectoryData:
    def __init__(self, init_state):
        self.inital_state = init_state

        # Going to store sequence of (state, action, reward, next_state, done)
        self.data_pairs = []

    def add_iteration(self, action, reward, next_state, done):
        if len(self.data_pairs) == 0:
            self.data_pairs.append((self.inital_state, action, reward, next_state, done))
        else:
            self.data_pairs.append((self.data_pairs[-1][3], action, reward, next_state, done))


class DataReader(Dataset):
    all_traj_data = []

    def __init__(self, read_file):
        super().__init__()
        self.all_traj_data = []
    
        # Processes the entire file and stores it into the all_traj_data field
        with h5py.File(read_file) as file:
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
    
    def __len__(self):
        return len(self.all_traj_data)
    
    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.all_traj_data[idx] 
        
        # Nomralize image data to between 0 and 1, also have shape (channels, height, width)
        state = torch.from_numpy(state).permute(2, 0, 1).float() / 255.0
        next_state = torch.from_numpy(next_state).permute(2, 0, 1).float() / 255.0
        action = torch.tensor(action).short()
        reward = torch.tensor(reward).float()
        done = torch.tensor(done).bool()

        return state, action, reward, next_state, done


if __name__ == "__main__":
    # Test with file from data_collection.py
    TEST_OUTPUT_FILENAME = "test_traj_long.h5"

    reader = DataReader(TEST_OUTPUT_FILENAME)

    print("Number of data tuples: ", len(reader.all_traj_data))

    print("First action: ", reader.all_traj_data[0][1])
    print("First reward: ", reader.all_traj_data[0][2])
    print("First done value: ", reader.all_traj_data[0][4])
    print("Shape of state: ", reader.all_traj_data[0][0].shape)
    print("Shape of next state: ", reader.all_traj_data[0][3].shape)

    # Test torch support
    dataloader = DataLoader(reader, batch_size=2, shuffle=True)
    
    for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(dataloader):
        print("all state in batch shape: ", states.shape)
        print("batch first state shape: ", states[0].shape)
        print("batch first next state shape: ", next_states[0].shape)
        
        print("actions: ", actions)
        print("rewards: ", rewards)
        print("dones: ", dones)

        # Only test one iteration
        break
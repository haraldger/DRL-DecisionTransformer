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

    def add_iteration(self, action, reward, next_state, reward_to_go, timestep, done):
        if len(self.data_pairs) == 0:
            self.data_pairs.append((self.inital_state, action, reward, next_state, reward_to_go, timestep, done))
        else:
            self.data_pairs.append((self.data_pairs[-1][3], action, reward, next_state, reward_to_go, timestep, done))


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
                read_reward_to_go = episode["reward_to_go"][()]
                read_timestep = episode["timestep"][()]

                if len(read_actions) == 0:
                    continue

                # Process data into trajectory pairs
                traj_data = TrajectoryData(read_states[0])
                for i in range(0, len(read_actions)):
                    traj_data.add_iteration(read_actions[i], read_rewards[i], read_states[i+1], read_reward_to_go[i], read_timestep[i], read_done[i])
                
                # Add epsiode trajectory pairs to all pairs
                self.all_traj_data += traj_data.data_pairs        
    
    def __len__(self):
        return len(self.all_traj_data)
    
    def __getitem__(self, idx):
        state, action, reward, next_state, reward_to_go, timestep, done = self.all_traj_data[idx] 
        
        # Nomralize image data to between 0 and 1, also have shape (channels, height, width)
        state = torch.from_numpy(state).permute(2, 0, 1).float() / 255.0
        next_state = torch.from_numpy(next_state).permute(2, 0, 1).float() / 255.0
        action = torch.tensor(action).short().unsqueeze(-1)
        reward = torch.tensor(reward).float().unsqueeze(-1)
        reward_to_go = torch.tensor(reward_to_go).float().unsqueeze(-1)
        timestep = torch.tensor(timestep).short().unsqueeze(-1)
        done = torch.tensor(done).bool().unsqueeze(-1)

        return state, action, reward, next_state, reward_to_go, timestep, done


def run_tests():
    # Test with file from data_collection.py
    TEST_OUTPUT_FILENAME = "test_traj_long.h5"

    reader = DataReader(TEST_OUTPUT_FILENAME)

    print("Number of data tuples: ", len(reader.all_traj_data))

    print("First action: ", reader.all_traj_data[0][1])
    print("First reward: ", reader.all_traj_data[0][2])
    print("First done value: ", reader.all_traj_data[0][6])
    print("First reward to go: ", reader.all_traj_data[0][4])
    print("First timestep: ", reader.all_traj_data[0][5])
    print("Shape of state: ", reader.all_traj_data[0][0].shape)
    print("Shape of next state: ", reader.all_traj_data[0][3].shape)

    # Test torch support
    dataloader = DataLoader(reader, batch_size=2, shuffle=False)
    
    for batch_idx, (states, actions, rewards, next_states, rewards_to_go, timesteps, dones) in enumerate(dataloader):
        print("all state in batch shape: ", states.shape)
        print("batch first state shape: ", states[0].shape)
        print("batch first next state shape: ", next_states[0].shape)
        
        print("actions: ", actions)
        print("rewards: ", rewards)
        print("rewards to go: ", rewards_to_go)
        print("timesteps: ", timesteps)
        print("dones: ", dones)

        print("actions shape: ", actions.shape)
        print("timesteps shape: ", timesteps.shape)
        print("rewards_to_go shape: ", rewards_to_go.shape)

        # Since only pos rewards, reward_to_go should be non-increasing 
        if rewards_to_go[0] < rewards_to_go[1]:
            print("Error: increasing reward to go")
            sys.exit() 
        # First reward to go should be different than first by only the single reward
        if rewards_to_go[0] - rewards[1] != rewards_to_go[1]:
            print("Error: Didn't decrement reward to go properly")
            sys.exit() 

        if timesteps[0] != 0 or timesteps[1] != 1:
            print("Error: in timesteps")
            sys.exit()
        # Only test one iteration
        break

    # Get the last iteration
    last_state_sample, last_action_sample, last_reward_sample, last_next_state_sample, last_rewards_to_go, last_timestep, last_done_sample = reader[-1]
    # Make sure the reward to go is 0
    if last_rewards_to_go != 0:
        print("Error with last reward to go: ", last_rewards_to_go)
        sys.exit() 
    
    if last_done_sample != True:
        print("Error last done sample: ", last_done_sample)
        sys.exit() 
    
    print("Passed data read test!")


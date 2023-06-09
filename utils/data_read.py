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
from utils.data_transforms import image_transformation, image_transformation_no_norm, image_transformation_crop_downscale, image_transformation_just_norm, image_transformation_grayscale_crop_downscale_norm, image_transformation_grayscale_crop_downscale

class TrajectoryData:
    def __init__(self, init_state):
        self.inital_state = init_state

        # Going to store sequence of (state, action, reward, next_state, done)
        self.data_pairs = []

        self.num_iterations = 0

    def add_iteration(self, action, reward, next_state, reward_to_go, timestep, done):
        if len(self.data_pairs) == 0:
            self.data_pairs.append((self.inital_state, action, reward, next_state, reward_to_go, timestep, done))
            self.num_iterations += 1
        else:
            self.data_pairs.append((self.data_pairs[-1][3], action, reward, next_state, reward_to_go, timestep, done))
            self.num_iterations += 1

    def fetch_last_k(self, k):
        # Return a list of the last k iterations
        # If there are not k iterations, duplicate the last one to pad out
        if self.num_iterations < k:
            return self.data_pairs + ([self.data_pairs[-1]] * (k-self.num_iterations))
        else:
            return self.data_pairs[-k:]
    
    def fetch_first_k(self, k):
        # Returns a list of the first k iterations
        # If there are not k iterations, duplicate the last one to pad out
        if self.num_iterations < k:
            return self.data_pairs + ([self.data_pairs[-1]] * (k-self.num_iterations))
        else:
            return self.data_pairs[:k]

class DataReader(Dataset):
    all_traj_data = []

    def __init__(
            self, 
            read_file, 
            k_last_iters=1000,
            k_first_iters=None, 
            store_transform=None, 
            return_transformation=False, 
            store_float_state=False, 
            return_float_state=False, 
            verbose_freq=None, 
            max_ep_load=None,
            debug_print=False
    ):
        super().__init__()
        self.all_traj_data = []
        
        # When fetching a traj, will get the last k iterations of it
        self.k_last_iters = k_last_iters

        # If a k_first_iters was given, use that instead
        self.k_first_iters = k_first_iters
        self.use_first_k = False
        if self.k_first_iters is not None:
            self.use_first_k = True

        # If passed, will use transformation on the state
        self.store_transform = store_transform
        self.store_float_state = store_float_state

        self.return_transform = return_transformation
        self.return_float_state = return_float_state

        trajectory_lengths = []

        # Processes the entire file and stores it into the all_traj_data field
        with h5py.File(read_file) as file:
            episodes = file.keys()
            if max_ep_load is None:
                total_num_ep = len(episodes)
            else:
                total_num_ep = min(max_ep_load, len(episodes))

            # pick total_num_ep random episodes
            load_episodes = random.sample(episodes, total_num_ep)

            for episode_num, ep in enumerate(load_episodes):
                if verbose_freq is not None and episode_num % verbose_freq == (verbose_freq-1):
                    print("Processing state: {}/{}".format(episode_num+1, total_num_ep))
        
                episode = file[ep]
                
                # Read data from episode
                read_states_compressed = episode["states"][:]
                read_states = []
                for i in range(read_states_compressed.shape[0]):
                    read_states.append(cv.imdecode(np.frombuffer(read_states_compressed[i], dtype=np.uint8), cv.IMREAD_UNCHANGED))

                if self.store_float_state:
                    read_states = torch.from_numpy(np.array(read_states)).permute(0, 3, 1, 2).float()
                else:
                    read_states = torch.from_numpy(np.array(read_states)).permute(0, 3, 1, 2)

                if self.store_transform is not None:
                    read_states = self.store_transform(read_states)

                read_actions = episode["actions"][()]
                read_rewards = episode["rewards"][()]
                read_done = episode["done"][()]
                read_reward_to_go = episode["reward_to_go"][()]
                read_timestep = episode["timestep"][()]

                trajectory_lengths.append(read_actions.shape[0])

                if len(read_actions) == 0:
                    continue

                # Process data into trajectory pairs
                traj_data = TrajectoryData(read_states[0])
                for i in range(0, len(read_actions)):
                    traj_data.add_iteration(read_actions[i], read_rewards[i], read_states[i+1], read_reward_to_go[i], read_timestep[i], read_done[i])
                
                self.all_traj_data.append(traj_data)   

        # If debug print, print out some stats on the trajectory data
        if debug_print:
            print("Average trjectory length: ", np.mean(trajectory_lengths))
            print("Median trajectory length: ", np.median(trajectory_lengths))
            print("Max trajectory length: ", np.max(trajectory_lengths))
            print("Min trajectory length: ", np.min(trajectory_lengths))

    def __len__(self):
        return len(self.all_traj_data)
    
    def __getitem__(self, idx):
        traj_data = self.all_traj_data[idx]
        if self.use_first_k:
            traj_pairs = traj_data.fetch_first_k(self.k_first_iters)
        else:
            traj_pairs = traj_data.fetch_last_k(self.k_last_iters)

        # States already stored transformed, so just stack them and apply return transform
        states = torch.stack([t[0] for t in traj_pairs])

        if self.return_float_state:
            states = states.float()

        if self.return_transform is not None:
            states = self.return_transform(states)

        # DT model doesn't even use next_states, so just don't reutrn them 
        # next_states = torch.tensor([t[3] for t in traj_pairs]).permute(0, 3, 1, 2).float() / 255.0
        
        actions = torch.tensor([t[1] for t in traj_pairs]).short().unsqueeze(-1)
        rewards = torch.tensor([t[2] for t in traj_pairs]).short().unsqueeze(-1)
        rewards_to_go = torch.tensor([t[4] for t in traj_pairs]).float().unsqueeze(-1)
        timesteps = torch.tensor([t[5] for t in traj_pairs]).int().unsqueeze(-1)
        dones = torch.tensor([t[6] for t in traj_pairs]).float().unsqueeze(-1)        
        
        return states, actions, rewards, rewards_to_go, timesteps, dones
        # return states, actions, rewards, next_states, rewards_to_go, timesteps, dones


def run_tests():
    # Test with file from data_collection.py
    TEST_OUTPUT_FILENAME = "test_traj_long.h5"

    reader = DataReader(
        TEST_OUTPUT_FILENAME, 
        store_transform=image_transformation_grayscale_crop_downscale, 
        store_float_state=False, 
        return_transformation=image_transformation_just_norm,
        return_float_state=True
    )

    print("Number of data trajectories: ", len(reader.all_traj_data))

    # Test torch support
    dataloader = DataLoader(reader, batch_size=2, shuffle=True)
    
    for batch_idx, (states, actions, rewards, rewards_to_go, timesteps, dones) in enumerate(dataloader):
        print("all state in batch shape: ", states.shape)
        print("batch first state seq shape: ", states[0].shape)
        
        print("state data type: ", type(states[0,0,0,0,0].item()))
        print("data: ", states[0,0,0,0,0])

        print("actions: ", actions[:,0:4])
        print("rewards: ", rewards[:,0:4])
        print("rewards to go: ", rewards_to_go[:,0:4])
        print("timesteps: ", timesteps[:,0:4])
        print("dones: ", dones[:,0:4])


        print("actions shape: ", actions.shape)
        print("timesteps shape: ", timesteps.shape)
        print("rewards_to_go shape: ", rewards_to_go.shape)

        # Since only pos rewards, reward_to_go should be non-increasing 
        if rewards_to_go[0][0] < rewards_to_go[0][1]:
            print("Error: increasing reward to go")
            sys.exit() 
        # First reward to go should be different than first by only the single reward
        if rewards_to_go[0][0] - rewards[0][1] != rewards_to_go[0][1]:
            print("Error: Didn't decrement reward to go properly")
            sys.exit() 

        if timesteps[0][0] != 0 or timesteps[0][1] != 1:
            print("Error: in timesteps")
            sys.exit()
    
        last_rewards_to_go = rewards_to_go[0][-1]
        if last_rewards_to_go != 0:
            print("Error with last reward to go: ", last_rewards_to_go)
            sys.exit() 

        last_done_sample = dones[0][-1]
        if last_done_sample != True:
            print("Error last done sample: ", last_done_sample)
            sys.exit() 

        # Only test one iteration
        break


    print("Passed data read test!")


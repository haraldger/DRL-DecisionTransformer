import sys
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from utils import constants

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayBuffer:
    def __init__(self, capacity=constants.REPLAY_MEMORY_SIZE, dims=constants.DIMENSIONS):
        """
        Replay memory for DQN.
        Stores state, action, next state and reward tuples.
        Capacity is the maximum number of tuples that can be stored.
        dims is the dimensions of the state space. This should be passed as a 
        tuple in the same form as the state space returned by gym environment.
        """
        self.capacity = capacity
        self.counter = 0
        self.length = 0
        self.dims = (dims[2], dims[0], dims[1]) # (C, H, W), PyTorch format
        self.state_memory_dims = (self.capacity, self.dims[0], self.dims[1], self.dims[2])

        # Initialize memories
        self.state_memory = torch.empty(size=self.state_memory_dims).type(torch.int8)
        self.action_memory = torch.empty(size=(self.capacity,1)).type(torch.int8)
        self.next_state_memory = torch.empty(size=self.state_memory_dims).type(torch.int8)
        self.reward_memory = torch.empty(size=(self.capacity,1)).type(torch.int8)
        self.done_memory = torch.empty(size=(self.capacity,1)).type(torch.bool)


    def add(self, state, action, next_state, reward, done):
        """
        Replay memory takes state, action, next state and reward tuples.
        All states are passed in their raw form as they are returned by gym environemnt.
        """

        # Add to memory at current index
        idx = int(self.counter % self.capacity)
        self.state_memory[idx] = torch.from_numpy(state).permute(2,0,1).type(torch.int8)
        self.action_memory[idx] = torch.Tensor([action]).type(torch.int8)
        self.next_state_memory[idx] = torch.from_numpy(next_state).permute(2,0,1).type(torch.int8)
        self.reward_memory[idx] = torch.Tensor([reward]).type(torch.int8)
        self.done_memory[idx] = torch.Tensor([False]).type(torch.bool)

        # Bookkeeping
        self.counter += 1
        self.counter = int(self.counter % self.capacity)
        self.length = min(self.length+1, self.capacity)
    

    def sample_tensor_batch(self, batch_size):
        """ 
        Returns a random sample of batch_size from the replay memory.
        """
        sample_indices = np.random.choice(self.length, batch_size)

        state_sample = self.state_memory[sample_indices].to(torch.float32)
        action_sample = self.action_memory[sample_indices].to(torch.long)
        next_state_sample = self.next_state_memory[sample_indices].to(torch.float32)
        reward_sample = self.reward_memory[sample_indices].to(torch.long)
        done_sample = self.done_memory[sample_indices]

        return state_sample, action_sample, next_state_sample, reward_sample, done_sample

    def show(self):
        print(".....")
        print("\nCapacity- ", self.capacity)
        print("\nCurrent length- ", self.length)
        print(".....")

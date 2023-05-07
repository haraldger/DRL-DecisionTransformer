import numpy as np
import math
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from Agents.agent import Agent
from networks.resnet import resnet34, resnet50 
from networks.tranformer import DecisionTransformer
import gym
from utils.data_load_transform import image_transformation, image_transformation_no_norm
from collections import deque
from torch.autograd.profiler import profile, record_function

class DTAgent(Agent):
    def __init__(
            self,
            env,
            config,
            num_blocks=8, 
            num_heads=8, 
            embedding_dim=384, 
            dropout=0.1, 
            *args,
            **kwargs
    ) -> None:
        super(DTAgent, self).__init__(env, config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.act_dim = env.action_space.n
        self.config = config
        self.max_ep_len = config["max_episode_length"]

        self.model = DecisionTransformer(
            num_blocks,
            num_heads,
            embedding_dim,
            dropout,
            self.max_ep_len,
            act_dim=self.act_dim,
            *args,
            **kwargs           
        )
    
        self.model = self.model.to(self.device)
        

    def cross_entropy_loss(self, action_preds, actions):
        # compute negative log-likelihood loss
        return F.cross_entropy(action_preds, actions)
    

    def train(
            self, 
            dataset, 
            batch_size,
            num_epochs,
            print_freq=5
    ):
        self.model.train()

        learning_rate = self.config["learning_rate"]

        # Training offline with expert tracjectories
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for batch_idx, (states, actions, rewards, returns_to_go, timesteps, dones) in enumerate(train_loader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                actions = actions.to(torch.long)
                returns_to_go = returns_to_go.to(self.device)
                timesteps = timesteps.to(self.device)
                timesteps = timesteps.to(torch.long)

                optimizer.zero_grad()
                a_preds = self.model.forward(states, actions, returns_to_go, timesteps).reshape(-1, self.act_dim)
                loss = self.cross_entropy_loss(a_preds, actions.reshape(-1))
                loss.backward()
                optimizer.step()

                if batch_idx % print_freq == (print_freq-1):
                    print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))

                del states, actions, returns_to_go, timesteps, a_preds, loss
                print(0/0)
            

    def predict_next_action(self, state_seq, action_seq, return_to_go_seq, timestep_seq):
        """ 
        Parameters:
            - state_seq - torch tensor of images (states)
                - Expected input shape: (batch_size, seq_length, channels, y, x)
                - Expects single channel image.
            - action_seq - torch tensor of actions. (shorts)
                - Expected input shape: (batch_size, seq_length, 1)
            - return_to_go_seq - torch tensor of returns to go (floats)
                - Expected input shape: (batch_size, seq_length, 1)
            - timestep_seq - what timestep you were on (shorts)
                - Expeected input shape: (batch_size, seq_legnth, 1)
        Precondition:
            - If this is the first step in evaluation, assumed that a start token has already been made
        """        
        state_seq = state_seq.to(self.device)
        action_seq = action_seq.to(self.device)
        return_to_go_seq = return_to_go_seq.to(self.device)
        timestep_seq = timestep_seq.to(self.device)

        self.model.eval()
        with torch.no_grad():
            action = self.model.forward(state_seq, action_seq, return_to_go_seq, timestep_seq)

        del state_seq, action_seq, return_to_go_seq, timestep_seq
        return action


    def run_evaluation_traj(self, target_reward=11000, traj_mem_size=1000, data_collection_obj=None, data_transformation=None, float_state=False):
        """ 
        Run a trajectory, predicting actions with the model.

        Parameters:
            - target_reward - what we want the reward to be 
                - Experiment with this value, max possible is 11000
            - traj_mem_size - how many of the past iterations we pass into model 
                - Another hyperparameter to tune
            - data_collection_obj - If not none, will collect trajectory information
            - data_transformation - If passed, will use this function to transform the data
            - float_state - If true, will convert state to float
        Returns:
            - reward: final reward of the trajectory
            - 
        """     

        state, _ = self.env.reset()
        y, x, z = state.shape
        inactive_frams = 65

        self.model.eval()

        for _ in range(inactive_frams):
            state, reward, done, info, _ = self.env.step(0)

        if data_collection_obj is not None:
            data_collection_obj.set_init_state(state)

        # Transform the state
        if float_state:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).float()
        else:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)

        if data_transformation is not None:
            state = data_transformation(state)

        # Create start token (using nop action)
        # Note: use of deque will ensure that we keep the most recent elements (only traj_mem_size)
        return_to_go_seq = deque(maxlen=traj_mem_size)
        return_to_go_seq.append(target_reward)
        state_seq = deque(maxlen=traj_mem_size)
        state_seq.append(state)
        action_seq = deque(maxlen=traj_mem_size)
        action_seq.append(0)
        timestep_seq = deque(maxlen=traj_mem_size)
        timestep_seq.append(0)

        seq_length = 1

        while not done:
            return_to_go_seq_torch = torch.tensor(return_to_go_seq).float().reshape(1, seq_length, 1)
            state_seq_torch = torch.stack(list(state_seq)).reshape(1, seq_length, 1, y, x)
            action_seq_torch = torch.tensor(action_seq).int().reshape(1, seq_length, 1)
            timestep_seq_torch = torch.tensor(timestep_seq).int().reshape(1, seq_length, 1)

            with torch.no_grad():
                next_action_pred = self.predict_next_action(state_seq_torch, action_seq_torch, return_to_go_seq_torch, timestep_seq_torch)
            
            next_action = torch.argmax(next_action_pred)
            
            next_state, reward, done, info, _ = self.env.step(next_action)

            if data_collection_obj is not None:
                data_collection_obj.store_next_step(next_action, reward, next_state, done)

            # Transform next state
            if float_state:
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0).float()
            else:
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)

            if data_transformation is not None:
                next_state = data_transformation(next_state)

            # update sequences
            return_to_go_seq.append(return_to_go_seq[-1] - reward)
            state_seq.append(next_state)
            action_seq.append(next_action)
            timestep_seq.append(timestep_seq[-1] + 1)
            
            # Keep only the last mem-length iterations
            # Implied by deque object

            seq_length += 1
        
        return reward, seq_length-1
            

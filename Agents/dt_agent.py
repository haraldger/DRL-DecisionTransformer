import numpy as np
import math
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from Agents.agent import Agent
from networks.resnet import resnet34, resnet50 
from networks.tranformer import DecisionTransformer
import gym

class DTAgent(Agent):
    def __init__(
            self,
            env,
            num_blocks=12, 
            num_heads=12, 
            embedding_dim=768, 
            dropout=0.1, 
            max_ep_len=10000, 
            *args,
            **kwargs
    ) -> None:
        
        self.act_dim = env.action_space.n
        self.max_ep_len = max_ep_len

        super().__init__(env, *args, **kwargs)
        self.model = DecisionTransformer(
            num_blocks,
            num_heads,
            embedding_dim,
            dropout,
            max_ep_len,
            act_dim=self.act_dim,
            *args,
            **kwargs           
        )

    def cross_entropy_loss(self, action_preds, actions):
        # compute negative log-likelihood loss
        return F.binary_cross_entropy(action_preds, actions)
    
    def train(
            self, 
            dataset, 
            batch_size,
            learning_rate=0.01,
            print_freq=5
    ):
        # Training offline with expert tracjectories
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, (states, actions, rewards, next_states, returns_to_go, timesteps, dones) in enumerate(train_loader):
            optimizer.zero_grad()
            a_preds = self.model.forward(states, actions, returns_to_go, timesteps)
            loss = self.cross_entropy_loss(a_preds, one_hot_actions)
            loss.backward()
            optimizer.step()

            if batch_idx % print_freq == (print_freq-1):
                print('Batch [{}/{}], Loss: {:.4f}'.format(
                    batch_idx+1, len(train_loader), loss.item()))

    def predict_next_action(self, state_seq, action_seq, return_to_go_seq, timestep_seq):
        """ 
        Parameters:
            - state_seq - torch tensor of images (states)
                - Expected input shape: (batch_size, channels, y, x)
            - action_seq - torch tensor of actions. (shorts)
                - Expected input shape: (batch_size, 1)
            - return_to_go_seq - torch tensor of returns to go (floats)
                - Expected input shape: (batch_size, 1)
            - timestep_seq - what timestep you were on (shorts)
                - Expeected input shape: (batch_size, 1)
        Precondition:
            - If this is the first step in evaluation, assumed that a start token has already been made
        """        
        action = self.model.forward(state_seq, action_seq, return_to_go_seq, timestep_seq)
        return action

    def run_evaluation_traj(self, target_reward=11000, traj_mem_size=1000, data_collection_obj=None):
        """ 
        Run a trajectory, predicting actions with the model.

        Parameters:
            - target_reward - what we want the reward to be 
                - Experiment with this value, max possible is 11000
            - traj_mem_size - how many of the past iterations we pass into model 
                - Another hyperparameter to tune
            - data_collection_obj - If not none, will collect trajectory information
        
        Returns:
            - reward: final reward of the trajectory
            - 
        """     

        state, _ = self.env.reset()
        inactive_frams = 65

        for _ in range(inactive_frams):
            state, reward, done, info, _ = self.env.step(0)

        if data_collection_obj is not None:
            data_collection_obj.set_init_state(state)

        # Create start token (using nop action)
        return_to_go_seq = [target_reward]
        state_seq = [state / 255.0]
        action_seq = [0]
        timestep_seq = [0]

        while not done:
            return_to_go_seq_torch = torch.tensor(return_to_go_seq).float().unsqueeze(-1)
            state_seq_torch = torch.tensor(state_seq).float()
            action_seq_torch = torch.tensor(action_seq).short().unsqueeze(-1)
            timestep_seq_torch = torch.tensor(timestep_seq).short().unsqueeze(-1)

            next_action_pred = self.predict_next_action(state_seq_torch, action_seq_torch, return_to_go_seq_torch, timestep_seq_torch)
            next_action = torch.argmax(next_action_pred)
            next_state, reward, done, info, _ = self.env.step(next_action)

            if data_collection_obj is not None:
                data_collection_obj.store_next_step(next_action, reward, next_state, done)

            # update sequences
            return_to_go_seq.append(return_to_go_seq[-1] - reward)
            state_seq.append(next_state / 255.0)
            action_seq.append(next_action)
            timestep_seq.append(timestep_seq[-1] + 1)
            
            # Keep only the last mem-length iterations
            return_to_go_seq[-traj_mem_size:]
            state_seq[-traj_mem_size:]
            action_seq[-traj_mem_size:]
            timestep_seq[-traj_mem_size:]
        
        return reward, timestep_seq[-1]
            

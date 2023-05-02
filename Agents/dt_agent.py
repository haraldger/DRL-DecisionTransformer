import numpy
import math
import torch
from torch import nn
from torch.optim import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from .agent import Agent
from ..networks.resnet import resnet34, resnet50 
from ..networks.tranformer import DecisionTransformer

class DTAgent(Agent):
    def __init__(
            self,
            num_blocks=12, 
            num_heads=12, 
            embedding_dim=768, 
            dropout=0.1, 
            max_ep_len=10000, 
            act_dim=9,
            *args,
            **kwargs
    ) -> None:
        
        super().__init__(*args, **kwargs)
        self.model = DecisionTransformer(
            num_blocks,
            num_heads,
            embedding_dim,
            dropout,
            max_ep_len,
            act_dim=act_dim,
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
            one_hot_actions = F.one_hot(actions, num_classes=self.act_dim)
            optimizer.zero_grad()
            a_preds = self.model.forward(states, one_hot_actions, returns_to_go, timesteps)
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
            - action_seq - torch tensor of actions (one-hot rep)
            - return_to_go_seq - torch tensor of returns to go (floats)
            - timestep_seq - what timestep you were onn (shorts)
        Precondition:
            - If this is the first step in evaluation, assumed that a start token has already been made
        """

        action = self.model.forward(state_seq, action_seq, return_to_go_seq, timestep_seq)
        return action

    def run_evaluation_traj(self, env, target_reward=11000, traj_mem_size=1000, data_collection_obj=None):
        """ 
        Run a trajectory, predicting actions with the model.

        Parameters:
            - env - AI Gym Ms Pac Man environment
            - target_reward - what we want the reward to be 
                - Experiment with this value, max possible is 11000
            - traj_mem_size - how many of the past iterations we pass into model 
                - Another hyperparameter to tune
            - data_collection_obj - If not none, will collect trajectory information
        """     

        state, _ = env.reset()
        inactive_frams = 65

        for _ in range(inactive_frams):
            state, reward, done, info, _ = env.step(0)

        data_collection_obj.set_init_state(state)

        # Create start token (using nop action)
        return_to_go_seq = [target_reward]
        state_seq = [state / 255.0]
        action_seq = [0]
        timestep_seq = [0]

        while not done:
            return_to_go_seq_torch = torch.tensor(return_to_go_seq).float()
            state_seq_torch = torch.tensor(state_seq).float()
            action_seq_torch = torch.tensor(action_seq).short()
            timestep_seq_torch = torch.tensor(timestep_seq).short()

            next_action_pred = self.predict_next_action(state_seq_torch, action_seq_torch, return_to_go_seq_torch, timestep_seq_torch)
            next_action = torch.argmax(next_action_pred)
            next_state, reward, done, info, _ = env.step(next_action)

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
        

        
            

def test_forward_pass():
    dt_model = DTAgent()

if __name__ == "__main__":
    test_forward_pass()

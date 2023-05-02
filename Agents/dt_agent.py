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
            *args,
            **kwargs           
        )

    def cross_entropy_loss(self, action_preds, actions):
        # compute negative log-likelihood loss
        return F.binary_cross_entropy(action_preds, actions)
    
    def predict_action(
            self, 
            states, 
            actions,  
            returns_to_go, 
            timesteps
    ):
        model_returns = self.model.forward(states, actions, returns_to_go, timesteps)
        return model_returns

    def train(
            self, 
            dataset, 
            batch_size,
            learning_rate=0.01,
            print_freq=5
    ):
        # Training offline with expert tracjectories
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, (states, actions, rewards, next_states, returns_to_go, timesteps, dones) in enumerate(train_loader):
            optimizer.zero_grad()
            a_preds = self.predict_action(states, actions, returns_to_go, timesteps)
            loss = self.cross_entropy_loss(a_preds, actions)
            loss.backward()
            optimizer.step()

            if batch_idx % print_freq == (print_freq-1):
                print('Batch [{}/{}], Loss: {:.4f}'.format(
                    batch_idx+1, len(train_loader), loss.item()))


def test_forward_pass():
    dt_model = DTAgent()

if __name__ == "__main__":
    test_forward_pass()

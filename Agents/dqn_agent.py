import numpy as np
import torch
from torch import nn
from .agent import Agent
from utils import experience_replay, epsilon_scheduler, constants

class DQNAgent(Agent):
    def __init__(self, env, replay_buffer=None, epsilon_scheduler=None):
        super(DQNAgent, self).__init__(env)
        self.target_net = DQN()
        self.policy_net = DQN()

        if replay_buffer is None:
            self.replay_buffer = experience_replay.ReplayBuffer()
        else:
            self.replay_buffer = replay_buffer

        if epsilon_scheduler is None:
            raise NotImplementedError
        else:
            self.epsilon_scheduler = epsilon_scheduler


    def act(self, state):
        return self.env.action_space.sample()
    
class DQN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        return self.layers(x)



class DQN_vanilla(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        return self.layers(x)




################## Unit tests ##################


def run_tests():
    test_dqn_forward_pass()
    test_dqn_vanilla_forward_pass()

def test_dqn_forward_pass():
    model = DQN()
    test_input = torch.randn(1, 3, 84, 84)
    print("DQN output shape: ", model(test_input).shape)

def test_dqn_vanilla_forward_pass():
    model = DQN_vanilla()
    test_input = torch.randn(1, 4, 84, 84)
    print("DQN_vanilla output shape: ", model(test_input).shape)
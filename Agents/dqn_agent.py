import numpy as np
import torch
from torch import nn
from .agent import Agent
from utils import experience_replay, epsilon_scheduler

class DQNAgent(Agent):
    def __init__(self, env, config, replay_buffer=None, scheduler=None):
        super(DQNAgent, self).__init__(env, config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iterations = 0
        self.training_mode = True

        # Initialize networks

        self.target_net = DQN().to(self.device)
        self.policy_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Hyperparameters

        self.learning_rate = self.config['learning_rate']
        self.gamma = self.config['gamma']

        # Initialize optimization objects

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

        # Initialize replay buffer and epsilon scheduler

        if replay_buffer is None:
            self.replay_buffer = experience_replay.ReplayBuffer()
        else:
            self.replay_buffer = replay_buffer

        if scheduler is None:
            self.scheduler = epsilon_scheduler.EpsilonScheduler()
        else:
            self.scheduler = scheduler


    def act(self, state):
        # Reshape state to (1, 3, 210, 160) PyTorch tensor
        torch_state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.training_mode == False: 
            with torch.no_grad():
                return self.policy_net(torch_state).argmax().item()

        if np.random.rand() < self.scheduler.get_epsilon() or self.iterations < self.config['initial_exploration']:
            action = self.env.action_space.sample()
            return action
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch_state)
                print(q_values)
                q_max = q_values.argmax().item()
                print(q_max)
                input()
                return q_max
    
    def train(self):
        """
        Perform one iteration of training.
        This function is called once per frame when training.
        """
        if self.iterations < self.config['initial_exploration']:
            self.iterations += 1
            return

        if self.iterations % self.config['dqn_update_frequency'] == 0:   # Train 
            state_sample, action_sample, next_state_sample, reward_sample, done_sample = self.replay_buffer.sample_tensor_batch(self.config['batch_size'], self.device)
            
            target_q_values = self.target_net(next_state_sample).max(1)[0].detach().view(-1, 1)
            targets = reward_sample + self.gamma * target_q_values * (1 - done_sample.long())

            preds = self.policy_net(state_sample).gather(1, action_sample)

            loss = self.loss(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.iterations % self.config['dqn_target_update_frequency'] == 0:   # Update target network
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.scheduler.step()
        self.iterations += 1

    def eval(self, value=True):
        """
        Set agent to evaluation mode. 
        Unlike train, this is a toggle rather than being called once per frame.
        """
        self.training_mode = not value
        if value == False:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def save(self, name):
        torch.save(self.policy_net.state_dict(), "results/" + name + ".pt")

    def load(self, name):
        self.policy_net.load_state_dict(torch.load("results/" + name + ".pt", map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
            
        
        

    
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22528, 1024),
            nn.ReLU(),
            nn.Linear(1024, 9)
        )

    def forward(self, x):
        return self.layers(x)



class DQN_vanilla(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22528, 512),
            nn.ReLU(),
            nn.Linear(512, 9)
        )

    def forward(self, x):
        return self.layers(x)




################## Unit tests ##################


def run_tests():
    test_dqn_forward_pass()
    test_dqn_vanilla_forward_pass()

def test_dqn_forward_pass():
    print("Running DQN forward pass test...")
    model = DQN()
    test_input = torch.randn(2, 3, 210, 160)
    print("Input shape: ", test_input.shape)
    print("Batch size: ", test_input.shape[0])
    y = model(test_input)
    print("DQN output shape: ", y.shape)
    assert y.shape == (2, 9)
    print("DQN forward pass test passed!")
    print()

def test_dqn_vanilla_forward_pass():
    print("Running DQN_vanilla forward pass test...")
    model = DQN_vanilla()
    test_input = torch.randn(2, 3, 210, 160)
    print("Input shape: ", test_input.shape)
    print("Batch size: ", test_input.shape[0])
    y = model(test_input)
    print("DQN_vanilla output shape: ", y.shape)
    assert y.shape == (2, 9)
    print("DQN_vanilla forward pass test passed!")
    print()
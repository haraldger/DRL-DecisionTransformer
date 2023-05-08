import numpy as np
import torch
from torch import nn
from .agent import Agent
from utils import experience_replay, epsilon_scheduler, data_transforms

class DQNAgent(Agent):
    def __init__(self, env, config, replay_buffer=None, scheduler=None):
        super(DQNAgent, self).__init__(env, config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iterations = 0
        self.training_mode = True

        # Initialize networks

        if self.config['dqn_network'] == 'vanilla':
            self.target_net = DQN_vanilla().to(self.device)
            self.policy_net = DQN_vanilla().to(self.device)
        elif self.config['dqn_network'] == 'large':
            self.target_net = DQN().to(self.device)
            self.policy_net = DQN().to(self.device)
        else:   
            raise ValueError("Invalid DQN network type. Valid types are 'vanilla' and 'large'.")
        
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

        
        # Performance monitoring

        self.last_100_q_values = []


    def act(self, state):
        """
        Choose an action to take given the current state.
        This function is called once per (stacked) frame when training.
        """
        if self.training_mode == False: 
            return self.exploit(state)

        if np.random.rand() < self.scheduler.get_epsilon() or self.iterations < self.config['initial_exploration']:
            action = self.env.action_space.sample()
            return action
        else:
            return self.exploit(state)
    
    def exploit(self, state):
        # Make state into PyTorch tensor
        torch_state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        torch_state = data_transforms.image_transformation_crop_downscale(torch_state)

        with torch.no_grad():
            out = self.policy_net(torch_state)  # Forward pass

            max_q = out.max().item()    # Performance monitoring
            self.last_100_q_values.append(max_q)
            if len(self.last_100_q_values) > 100:
                self.last_100_q_values.pop(0)

            action = out.argmax().item()
            return action

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

        # Second decay, slower than first and decays learning rate
        if self.config['second_decay'] and self.iterations == self.config['decay_frames']:
            self.scheduler = epsilon_scheduler.EpsilonScheduler(initial_epsilon=self.config['final_epsilon'], final_epsilon=0, decay_frames=self.config['decay_frames'], decay_mode=self.config['decay_mode'], decay_rate=self.config['decay_rate'], start_frames=0)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config['learning_rate'] / 10

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

    def epsilon(self):
        return self.scheduler.get_epsilon()

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
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(18432, 1024),
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
            nn.Linear(3136, 512),
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
    test_input = torch.randn(2, 3, 84, 84)
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
    test_input = torch.randn(2, 3, 84, 84)
    print("Input shape: ", test_input.shape)
    print("Batch size: ", test_input.shape[0])
    y = model(test_input)
    print("DQN_vanilla output shape: ", y.shape)
    assert y.shape == (2, 9)
    print("DQN_vanilla forward pass test passed!")
    print()
from .agent import Agent

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, state):
        return self.env.action_space.sample()
    
    def train(self, end_of_episode=False):
        pass

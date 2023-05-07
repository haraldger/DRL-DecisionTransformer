from .agent import Agent

class RandomAgent(Agent):
    def __init__(self, env, config):
        super().__init__(env, config)

    def act(self, state):
        return self.env.action_space.sample()
    
    def train(self, end_of_episode=False):
        pass

    def eval(self, eval_mode=True):
        pass

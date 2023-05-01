from .agent import Agent

class DQNAgent(Agent):
    def __init__(self, env):
        super(DQNAgent, self).__init__(env)


    def act(self, state):
        return self.env.action_space.sample()
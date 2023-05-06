class Agent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def act(self, state):
        raise NotImplementedError
    
    def train(self, end_of_episode=False):
        raise NotImplementedError
    
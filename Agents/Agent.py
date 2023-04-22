class Agent:
    def __init__(self, env):
        self.env = env

    def act(self, observation, reward, done):
        raise NotImplementedError
import Agent 

class DQNAgent(Agent):
    def __init__(self, env):
        super(DQNAgent, self).__init__(env)


    def act(self, observation, reward, done):
        raise NotImplementedError
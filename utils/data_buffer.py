class DataBuffer:
    """
    This class is a data buffer that temporarily stores data from a trajectory.
    Its intended use is to store data from a single trajectory, and then pass
    it to the real data collector to be written to disk if the trajectory is
    good enough.
    """

    def __init__(self, init_state, data_collector, threshold=500):
        """
        Initialize the data buffer with the initial state of the trajectory.
        """
        self.inital_state = init_state
        self.data_collector = data_collector
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.data_pairs = []
        self.episode_reward = 0

    def set_init_state(self, init_state):
        """
        Set the initial state of the trajectory.
        """
        self.inital_state = init_state

    def add_iteration(self, action, reward, next_state,  done):
        """
        Add a single iteration to the data buffer.
        """
        if len(self.data_pairs) == 0:
            self.data_pairs.append((self.inital_state, action, reward, next_state, done))
        else:
            self.data_pairs.append((self.data_pairs[-1][3], action, reward, next_state, done))

    def finalize(self):
        """
        Finalize the data buffer.
        Commits the data to the data collector if the episode reward is above
        the threshold.
        """
        if self.episode_reward < self.threshold:
            self.reset()
        else:
            print("Writing trajectory to data collector.")
            self.data_collector.set_init_state(self.inital_state)
            for data_pair in self.data_pairs:
                self.data_collector.store_next_step(*data_pair)
            self.reset()

            



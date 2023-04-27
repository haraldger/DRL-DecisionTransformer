class EpsilonScheduler:
    def __init__(self, initial_epsilon=1.0, final_epsilon=0.01, decay_frames=1E6, decay_mode='single', decay_rate=0.1, start_frames=0) -> None:
        """
        initial_epsilon (float):  Initial value of epsilon.
        final_epsilon (float): Final value of epsilon, will not decay beyond this.
        decay_frames (int): Number of frames after which to decay epsilon.
        decay_mode (string={'single', 'multiple'}): Mode of operation for decay. Single will perform a single decay
            after decay_frames to set epsilon to final_epsilon. Multiple will decay epsilon by decay_rate.
        decay_rate (float): The rate at which epsilon decays. E.g., if 0.1, epsilon will be divided by 10 after 
            every decay_frames number of frames, until it is less than or equal to final_epsilon. Useless if
            decay_mode == 'single'.
        """
        self.initial_epsilon = initial_epsilon
        self.decay_frames = decay_frames
        self.final_epsilon = final_epsilon
        self.decay_mode = decay_mode
        self.decay_rate = decay_rate
        self.start_frames = start_frames

        self.current_epsilon = self.initial_epsilon
        self.frames = 0

    def step(self):
        """
        Increment number of frames by 1.
        """
        self.frames += 1

        if self.frames < self.start_frames:
            return

        if self.decay_mode == 'multiple':
            if self.frames % self.decay_frames == 0:
                self.current_epsilon = max(self.final_epsilon, self.current_epsilon * self.decay_rate)

        else:
            if self.frames % self.decay_frames == 0:
                self.current_epsilon = self.final_epsilon

    def get_epsilon(self):
        return self.current_epsilon

        
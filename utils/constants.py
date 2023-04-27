PRINT_FREQUENCY = 1000          # Frequency in episodes to print progress
REPLAY_MEMORY_SIZE = 100000     # Size of experience replay memory
DIMENSIONS = (3, 210, 160)      # Dimensions of state space
INITIAL_EPSILON = 1.0           # Initial value of epsilon in epsilon-greedy exploration
FINAL_EPSILON = 0.01            # Final value of epsilon in epsilon-greedy exploration
DECAY_FRAMES = 1E6              # Number of frames after which to decay epsilon
DECAY_MODE = 'single'           # Mode of operation for decay. Single will perform a single decay after decay_frames to set epsilon to final_epsilon. Multiple will decay epsilon by decay_rate.
DECAY_RATE = 0.1                # The rate at which epsilon decays. E.g., if 0.1, epsilon will be divided by 10 after every decay_frames number of frames, until it is less than or equal to final_epsilon. Useless if decay_mode == 'single'.

def load():
    config = {
        'print_frequency': PRINT_FREQUENCY,
        'replay_memory_size': REPLAY_MEMORY_SIZE,
        'dimensions': DIMENSIONS,
        'initial_epsilon': INITIAL_EPSILON,
        'final_epsilon': FINAL_EPSILON,
        'decay_frames': DECAY_FRAMES,
        'decay_mode': DECAY_MODE,
        'decay_rate': DECAY_RATE,
    }
    return config
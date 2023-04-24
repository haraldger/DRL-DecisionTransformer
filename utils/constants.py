PRINT_FREQUENCY = 1000          # Frequency in episodes to print progress
REPLAY_MEMORY_SIZE = 100000     # Size of experience replay memory
DIMENSIONS = (3, 210, 160)      # Dimensions of state space

def load():
    config = {
        'print_frequency': PRINT_FREQUENCY,
        'replay_memory_size': REPLAY_MEMORY_SIZE
    }
    return config
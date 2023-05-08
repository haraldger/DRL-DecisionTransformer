PRINT_FREQUENCY = 1000                  # Frequency in episodes to print progress
MODEL_SAVE_FREQUENCY = 500             # Frequency in episodes to save model
EVALUATION_FREQUENCY = 500             # Frequency in episodes to evaluate model
REPLAY_MEMORY_SIZE = 100000             # Size of experience replay memory
DIMENSIONS = (210, 160, 3)              # Dimensions of state space as returned by gym environment
INITIAL_EPSILON = 1.0                   # Initial value of epsilon in epsilon-greedy exploration
FINAL_EPSILON = 0.1                    # Final value of epsilon in epsilon-greedy exploration
INITIAL_EXPLORATION = 50000             # Number of frames to perform random actions before starting training
DECAY_FRAMES = 2E6                      # Number of frames after which to decay epsilon
DECAY_MODE = 'linear'                   # Mode of operation for decay. Single will perform a single decay after decay_frames to set epsilon to final_epsilon. Multiple will decay epsilon by decay_rate.
DECAY_RATE = 0.99                        # The rate at which epsilon decays. E.g., if 0.1, epsilon will be divided by 10 after every decay_frames number of frames, until it is less than or equal to final_epsilon. Useless if decay_mode == 'single' or 'linear'.
LEARNING_RATE = 0.00025             # Learning rate for DQN
GAMMA = 0.99                            # Discount factor for future rewards
DQN_UPDATE_FREQUENCY = 4                # Number of actions taken between successive SGD updates
DQN_TARGET_UPDATE_FREQUENCY = 40000      # Number of actions taken between successive target network updates
BATCH_SIZE = 32                         # Batch size for Optimizer
MAX_EPISODE_LENGTH = 10000              # Maximum number of frames in an episode


# Constants for DT ----------------------------------------------
LEARNING_RATE_DT = 0.00001

# Number of Batches
MODEL_SAVE_FREQUENCY_DT = 500
EVALUATION_FREQUENCY_DT = 500

def load():
    config = {
        'print_frequency': PRINT_FREQUENCY,
        'model_save_frequency': MODEL_SAVE_FREQUENCY,
        'evaluation_frequency': EVALUATION_FREQUENCY,
        'replay_memory_size': REPLAY_MEMORY_SIZE,
        'dimensions': DIMENSIONS,
        'initial_epsilon': INITIAL_EPSILON,
        'final_epsilon': FINAL_EPSILON,
        'initial_exploration': INITIAL_EXPLORATION,
        'decay_frames': DECAY_FRAMES,
        'decay_mode': DECAY_MODE,
        'decay_rate': DECAY_RATE,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'dqn_update_frequency': DQN_UPDATE_FREQUENCY,
        'dqn_target_update_frequency': DQN_TARGET_UPDATE_FREQUENCY,
        'batch_size': BATCH_SIZE,
        'max_episode_length': MAX_EPISODE_LENGTH,
        'model_save_frequency_dt': MODEL_SAVE_FREQUENCY_DT,
        'evaluation_frequency_dt': EVALUATION_FREQUENCY_DT,
        'learning_rate_dt': LEARNING_RATE_DT
    }
    return config
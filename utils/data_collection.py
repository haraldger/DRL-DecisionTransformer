""" 
Purpose:    Collects data from an trained agent, and saves this data into a csv file.
"""
import sys
import numpy as np
import random
import h5py
import cv2 as cv

class DataCollector():
    def __init__(self, out):
        self.outfile_path = out
        self.episode_count = 0

        self.state_buff = []
        self.reward_buff = []
        self.action_buff = []

        # clear/restart file
        with h5py.File(self.outfile_path, 'w') as file:
            pass

    def set_init_state(self, state):
        self.state_buff = [cv.imencode('.png', state)[1].tobytes()]

    def store_next_step(self, action, reward, next_state, done):
        """ 
        Takes in parameters and updates the data collector's buffers.
        If the episode is done, will output values to outfile and reset buffers.
        """

        self.state_buff.append(cv.imencode('.png', next_state)[1].tobytes())
        self.reward_buff.append(reward)
        self.action_buff.append(action)
        
        if done:
            # Write data into the outfile
            # This will be in pairs of (state, action, reward, next_stete, done)
            num_iterations = len(self.action_buff)
            with h5py.File(self.outfile_path, 'a') as file:
                grp = file.create_group(f'episode_{self.episode_count}')
                grp.create_dataset(f'states', data=np.array(self.state_buff), compression='gzip')
                grp.create_dataset(f'actions', data=np.array(self.action_buff))
                grp.create_dataset(f'rewards', data=np.array(self.reward_buff))
                done_arr = np.full((num_iterations), False)
                done_arr[-1] = True
                grp.create_dataset(f'done', data=done_arr)
                
            # Reset buffers
            self.state_buff = []
            self.reward_buff = []
            self.action_buff = []

            self.episode_count += 1

def run_tests():
    # Used for testing data_collection functionality
    TEST_OUTPUT_FILENAME = "test_traj.h5"
    
    """ 
    AI Gym link: https://www.gymlibrary.dev/environments/atari/ms_pacman/
    Action space: Discrete(9) - if full_action_space=False
    Observation Space = (210, 160, 3) - values from 0 to 255
    """

    state_shape = (210, 160, 3)
    action_size = 9

    state0 = np.zeros(state_shape, dtype=np.uint8)
    # state1 = np.copy(state0)
    # state2 = np.copy(state0)
    state1 = np.random.randint(0, 256, size=state_shape, dtype=np.uint8)
    state2 = np.random.randint(0, 256, size=state_shape, dtype=np.uint8)
    states = [state0, state1, state2]

    actions = np.random.randint(0, action_size, size=(2))
    rewards = np.random.uniform(-5, 5, size=(2))
    done_vals = [False, True]

    collector = DataCollector(TEST_OUTPUT_FILENAME)
    collector.set_init_state(states[0])
    collector.store_next_step(actions[0], rewards[0], states[1], done_vals[0])
    collector.store_next_step(actions[1], rewards[1], states[2], done_vals[1])

    # Should be empty (reset)
    print("Should be empty: ", collector.action_buff)

    print("Reading from file ...")
    # Read from hdf5 file just to make sure it was stored correctly
    with h5py.File(TEST_OUTPUT_FILENAME, 'r') as file:
        for ep in file.keys():
            episode = file[ep]
            read_states_compressed = episode["states"][:]
            read_states = []
            for i in range(read_states_compressed.shape[0]):
                read_states.append(np.frombuffer(read_states_compressed[i], dtype=np.uint8))

            read_actions = episode["actions"][()]
            read_rewards = episode["rewards"][()]
            read_done = episode["done"][()]

            for i in range(0, len(read_actions)):
            
                if cv.imdecode(read_states[i], cv.IMREAD_UNCHANGED).tolist() != states[i].tolist():
                    print("Erorr: State Discrepency")
                    # print("Read State: \n", state)
                    # print("Actual State: \n", states[i].tolist())
                    sys.exit()
                
                if read_actions[i] != actions[i]:
                    print("Error: Action Discrepency")
                    print("Read action: ", read_actions[i])
                    print("Actual action: ", actions[i])
                    sys.exit()
                
                if read_rewards[i] != rewards[i]:
                    print("Error: Reward Discrepency")
                    print("Read reward: ", read_rewards[i])
                    print("Actual reward: ", rewards[i])
                    sys.exit()
                
                if cv.imdecode(read_states[i+1], cv.IMREAD_UNCHANGED).tolist() != states[i+1].tolist():
                    print("Erorr: Next State Discrepency")
                    # print("Read State: \n", next_state)
                    # print("Actual State: \n", states[i+1].tolist())
                    sys.exit()
                
                if read_done[i] != done_vals[i]:
                    print("Error: Done Discrepency")
                    print("Read done val: ", read_done[i])
                    print("Actual done val: ", done_vals[i])
                    sys.exit()
            
    print("Passed: All rows matched!")

    print("Large File Size Test")
    TEST_OUTPUT_FILENAME = "test_traj_long.h5"
    traj_length = 1000

    collector = DataCollector(TEST_OUTPUT_FILENAME)
    collector.set_init_state(state0)
    
    rand_states = []
    rand_actions = np.random.randint(0, action_size, size=(traj_length))
    rand_rewards = np.random.uniform(-5, 5, size=(traj_length))
    for i in range(0,traj_length):
        if i % 100 == 99:
            print("Iteration: ", i)
    
        rand_states.append(np.random.randint(0, 256, size=state_shape, dtype=np.uint8))
        collector.store_next_step(rand_actions[i], rand_rewards[i], rand_states[-1], (i==(traj_length-1)))

    print("Complete")


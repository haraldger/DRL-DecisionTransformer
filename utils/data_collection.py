""" 
Purpose:    Collects data from an trained agent, and saves this data into a csv file.
"""
import sys
import numpy as np
import random
import h5py
import cv2 as cv

class DataCollector():
    def __init__(self, out, episodes_per_write=1):
        self.outfile_path = out
        self.episode_count = 0

        self.state_buff = []
        self.reward_buff = []
        self.action_buff = []
        self.running_reward = []

        # Episodes are stored in write buffer until they need to be printed out 
        self.write_buffer = {}
        self.episodes_per_write = episodes_per_write

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

        if len(self.running_reward) == 0:
            self.running_reward = [reward]
        else:
            self.running_reward.append(self.running_reward[-1] + reward)
        
        if done:
            num_iterations = len(self.action_buff)
            # Check if we should write to file, or just write to buffer
            eps_in_buffer = self.write_buffer.keys()
            if len(eps_in_buffer) == (self.episodes_per_write-1):
                # Write data into the outfile (both this episode and buffer episodes)
                # This will be in pairs of (state, action, reward, next_state, reward_to_go, done)
                
                # create array for reward to go, based on the running_rewards
                reward_to_go = self.running_reward[-1] - np.array(self.running_reward)

                with h5py.File(self.outfile_path, 'a') as file:
                    grp = file.create_group(f'episode_{self.episode_count}')
                    grp.create_dataset(f'states', data=np.array(self.state_buff), compression='gzip')
                    grp.create_dataset(f'actions', data=np.array(self.action_buff))
                    grp.create_dataset(f'rewards', data=np.array(self.reward_buff))
                    grp.create_dataset(f'reward_to_go', data=np.array(reward_to_go))
                    grp.create_dataset(f'timestep', data=np.arange(len(self.action_buff)))
                    done_arr = np.full((num_iterations), False)
                    done_arr[-1] = True
                    grp.create_dataset(f'done', data=done_arr)
                
                    # Write buffer data to outfile
                    keys = self.write_buffer.keys()
                    for k in keys:
                        temp_data = self.write_buffer[k]

                        # create reward to go array
                        reward_to_go = temp_data["running_reward"][-1] - np.array(temp_data["running_reward"])

                        grp = file.create_group(f'episode_{k}')
                        grp.create_dataset(f'states', data=np.array(temp_data["state"]), compression='gzip')
                        grp.create_dataset(f'actions', data=np.array(temp_data["action"]))
                        grp.create_dataset(f'rewards', data=np.array(temp_data["reward"]))
                        grp.create_dataset(f'reward_to_go', data=np.array(reward_to_go))
                        grp.create_dataset(f'timestep', data=np.arange(len(temp_data["action"])))
                        grp.create_dataset(f'done', data=temp_data["done"])

                # Clear buffer
                self.write_buffer = {}

            else:
                # Write data to write buffer
                done_arr = np.full((num_iterations), False)
                done_arr[-1] = True
                self.write_buffer[self.episode_count] = {}
                self.write_buffer[self.episode_count]["state"] = self.state_buff
                self.write_buffer[self.episode_count]["action"] = self.action_buff
                self.write_buffer[self.episode_count]["reward"] = self.reward_buff
                self.write_buffer[self.episode_count]["running_reward"] = self.running_reward
                self.write_buffer[self.episode_count]["done"] = done_arr
                       
            # Reset buffers
            self.state_buff = []
            self.reward_buff = []
            self.action_buff = []
            self.running_reward = []

            self.episode_count += 1

    def dump_write_buffer(self):
        # write all data in write buffer to file
        with h5py.File(self.outfile_path, 'a') as file:
            keys = self.write_buffer.keys()
            for k in keys:
                temp_data = self.write_buffer[k]

                # create reward to go array
                reward_to_go = temp_data["running_reward"][-1] - np.array(temp_data["running_reward"])

                grp = file.create_group(f'episode_{k}')
                grp.create_dataset(f'states', data=np.array(temp_data["state"]), compression='gzip')
                grp.create_dataset(f'actions', data=np.array(temp_data["action"]))
                grp.create_dataset(f'rewards', data=np.array(temp_data["reward"]))
                grp.create_dataset(f'reward_to_go', data=np.array(reward_to_go))
                grp.create_dataset(f'timestep', data=np.arange(len(temp_data["action"])))
                grp.create_dataset(f'done', data=temp_data["done"])

        # Clear buffer
        self.write_buffer = {}

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
    rewards = np.random.randint(0, 5, size=(2))
    done_vals = [False, True]

    collector = DataCollector(TEST_OUTPUT_FILENAME, episodes_per_write=2)
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
            read_reward_to_go = episode["reward_to_go"][()]
            read_timestep = episode["timestep"][()]

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
                
                if (i == (len(read_actions)-1) and read_reward_to_go[i] !=0) or read_reward_to_go[i] != np.sum(rewards[i+1:]):
                    print("Error: Rewards to Go")
                    print("iteration: ", i)
                    print("Read Value: ", read_reward_to_go[i])
                    if i == (len(read_actions)-1):
                        print("Last iteration: should be 0")
                        sys.exit()
                    else:
                        print("Should be: ", np.sum(rewards[i+1:]))
                        print("Remaining rewards: ", rewards[i+1:])
                        sys.exit()

                if read_timestep[i] != i:
                    print("Error: Timestep")
                    print("iteration: ", i)
                    print("Read value: ", read_timestep[i])
                    sys.exit()

    print("Passed: All rows matched!")

    print("Large File Size Test")
    TEST_OUTPUT_FILENAME = "test_traj_long.h5"
    traj_length = 1000

    collector = DataCollector(TEST_OUTPUT_FILENAME)
    collector.set_init_state(state0)
    
    rand_states = []
    rand_actions = np.random.randint(0, action_size, size=(traj_length))
    rand_rewards = np.random.randint(0, 5, size=(traj_length))
    for i in range(0,traj_length):
        if i % 100 == 99:
            print("Iteration: ", i)
    
        rand_states.append(np.random.randint(0, 256, size=state_shape, dtype=np.uint8))
        collector.store_next_step(rand_actions[i], rand_rewards[i], rand_states[-1], (i==(traj_length-1)))

    print("Complete")


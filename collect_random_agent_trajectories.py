import sys
import argparse
import time
import gym
from utils import experience_replay, constants, data_collection
from Agents import dt_agent, random_agent, dqn_agent
import os

config = dict()
env = None

def run():
    # Global objects
    global config 
    global env

    # Reset environment
    state, _ = env.reset()
    inactive_frames = 65

    # Initialize objects
    agent = random_agent.RandomAgent(env)

    # Initalize data collector
    if os.path.isfile(config['output']):
        sys.exit("Ouput file path already exists.")
    dc = data_collection.DataCollector(config['output'], episodes_per_write=config['dump_frequency'])

    # loop
    for i in range(config['num_episodes']):
        episode_reward = 0

        # Skip inactive frames
        for _ in range(inactive_frames):
            state, reward, done, info, _ = env.step(0)

        dc.set_init_state(state)

        while not done:     # Run episode until done
            action = agent.act(state)
            next_state, reward, done, info, _ = env.step(action)
            episode_reward += reward
            
            dc.store_next_step(action, reward, next_state, done)

            state = next_state

        state, _ = env.reset()
        if config['verbose'] and i % config['print_frequency'] == 0:
            print('Episode: {}/{}. Reward: {}'.format(i, config['num_episodes'], episode_reward))
    
    dc.dump_write_buffer()
    env.close()


def main():
    # Set configuration
    global config
    # Load constants
    config.update(constants.load())

    # Initialize environment
    global env 
    env = gym.make('ALE/MsPacman-v5')

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Playing MsPacman with Reinforcement Learning agents.')
    parser.add_argument('-n', '--num_episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-o', '--output', type=str, default="random_trajectories.h5", help="Output Trajectory Path")
    parser.add_argument('-pf', '--print_frequency', type=int, help='Frequency in episodes to print progress')
    parser.add_argument('-df', '--dump_frequency', type=int, default=50, help="How many episodes between writing trajectories to outfile")
    args = parser.parse_args()

    config['num_episodes'] = args.num_episodes
    config['verbose'] = args.verbose
    config['output'] = args.output
    if args.print_frequency:
        config['print_frequency'] = args.print_frequency
    else:
        config['print_frequency'] = constants.PRINT_FREQUENCY
    
    config['dump_frequency'] = args.dump_frequency

    print("Print frequency is: ", config['print_frequency'])

    print('Playing MsPacman with random agent')
    print('Mode: evaluation/data collection')

    # Run
    run()

if __name__ == '__main__':
    main()
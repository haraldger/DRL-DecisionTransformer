import sys
import argparse
import time
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from utils import experience_replay, epsilon_scheduler, constants
from Agents import dt_agent, random_agent, dqn_agent
from utils.data_collection import DataCollector

config = dict()
agent = None
env = None
replay_buffer = None
scheduler = None


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Playing MsPacman with Reinforcement Learning agents.')
    parser.add_argument('-a', '--agent', choices=['random', 'dqn', 'dt'], type=str, default='random', help='Agent to use')
    parser.add_argument('-t', '--train', action='store_true', help='Train the agent')
    parser.add_argument('-n', '--num_episodes', type=int, default=100000, help='Number of episodes to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-pf', '--print_frequency', type=int, help='Frequency in episodes to print progress')
    parser.add_argument('-df', '--dump_frequency', type=int, default=None, help="How many episodes between writing trajectories to outfile")
    parser.add_argument('-o', '--output', type=str, default="random_trajectories.h5", help="Output trajectory path for data collection")
    parser.add_argument('--evaluation_frequency', type=int, help='Frequency in episodes to evaluate model')

    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate')
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor')
    parser.add_argument('-l', '--load', type=str, default="None", help='Load model. Provide name of model file, without extension or folder')
    parser.add_argument('--initial_epsilon', type=float, help='Initial epsilon for epsilon-greedy exploration')
    parser.add_argument('--final_epsilon', type=float, help='Final epsilon for epsilon-greedy exploration')
    parser.add_argument('--initial_exploration', type=int, help='Number of frames to perform random actions before starting training')

    args = parser.parse_args()

    # Set configuration
    global config
    config.update(constants.load())     # Load constants

    # Set configuration based on arguments
    config['agent'] = args.agent
    config['train'] = args.train
    config['num_episodes'] = args.num_episodes
    config['verbose'] = args.verbose
    config['load'] = args.load
    config['dump_frequency'] = args.dump_frequency
    config['output'] = args.output

    if args.print_frequency is not None:
        config['print_frequency'] = args.print_frequency

    if args.evaluation_frequency is not None:
        config['evaluation_frequency'] = args.evaluation_frequency

    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate

    if args.gamma is not None:
        config['gamma'] = args.gamma

    if args.initial_epsilon is not None:
        config['initial_epsilon'] = args.initial_epsilon

    if args.final_epsilon is not None:
        config['final_epsilon'] = args.final_epsilon

    if args.initial_exploration is not None:
        config['initial_exploration'] = args.initial_exploration
        

    print("Print frequency is: ", config['print_frequency'])

    config['experience_replay'] = False
    config['epsilon_scheduler'] = False
    config['save'] = False
    if args.agent == 'dqn' and args.train:
        config['experience_replay'] = True
        config['epsilon_scheduler'] = True
        config['save'] = True


    # Initialize environment
    global env 
    env = gym.make('ALE/MsPacman-v5')

    print('Playing MsPacman with {} agent'.format(args.agent))
    print('Mode: {}'.format('train' if args.train else 'evaluation'))

    # Run
    run()


def run():
    # Global objects
    global config 
    global agent
    global env
    global replay_buffer

    # Reset environment
    state, _ = env.reset()
    inactive_frames = 65
    total_frames = 0


    # Initialize objects
    global replay_buffer
    if config['experience_replay']:
        replay_buffer = experience_replay.ReplayBuffer(config['replay_memory_size'], config['dimensions'])

    global scheduler
    if config['epsilon_scheduler']:
        scheduler = epsilon_scheduler.EpsilonScheduler(config['initial_epsilon'], config['final_epsilon'], config['decay_frames'], config['decay_mode'], config['decay_rate'])

    global agent
    if config['agent'] == 'random':
        agent = random_agent.RandomAgent(env, config)

    elif config['agent'] == 'dqn':
        agent = dqn_agent.DQNAgent(env, config, replay_buffer, scheduler)
        
        if config['load'] != 'None':
            agent.load(config['load'])
        
        if not config['train']:
            agent.eval(True)

    elif config['agent'] == 'dt':
        agent = dt_agent.DTAgent(env, config)


    # Initalize data collector
    if config['dump_frequency'] is not None:
        if os.path.isfile(config['output']):
            sys.exit("Ouput file path already exists.")
        dc = DataCollector(config['output'], episodes_per_write=config['dump_frequency'])

    # Game loop
    last_100_rewards = []
    median_running_rewards = []
    median_evaluation_rewards = []
    for i in range(config['num_episodes']):
        episode_reward = 0
        

        # Skip inactive frames
        for _ in range(inactive_frames):
            state, reward, done, info, _ = env.step(0)

        if config['dump_frequency'] is not None:
            dc.set_init_state(state)

        while not done:     # Run episode until done
            action = agent.act(state)
            cumulative_reward = 0
            cumulative_state = state
            for _ in range(3):
                next_state, reward, done, info, _ = env.step(action)
                cumulative_reward += reward
                cumulative_state = np.maximum(cumulative_state, next_state)
                if done:
                    break

            episode_reward += cumulative_reward
            
            if config['experience_replay']:
                replay_buffer.add(state, action, cumulative_state, cumulative_reward, done)

            state = cumulative_state

            if config['dump_frequency'] is not None:
                dc.store_next_step(action, reward, next_state, done)
                
            if config['train']:
                agent.train()

            total_frames += 1

        # Monitoring performance
        last_100_rewards.append(episode_reward)
        if len(last_100_rewards) > 100:
            last_100_rewards.pop(0)
        median_running_rewards.append(np.median(last_100_rewards))

        state, _ = env.reset()
        if config['verbose'] and i % config['print_frequency'] == 0:
            print('Episode: {}/{}, total iterations: {}. Reward: {}. median running reward: {}'.format(i, config['num_episodes'], total_frames, episode_reward, np.median(last_100_rewards)))

        if config['save'] and i % config['model_save_frequency'] == 0 and i != 0:
            save_name = time.strftime("%Y%m%d-%H%M%S")
            save_name += '_episodes_' + str(i)
            agent.save(save_name)
            # Save performance graph
            plt.figure()
            plt.plot(range(len(median_running_rewards)), median_running_rewards)
            plt.xlabel('Episodes')
            plt.ylabel('median running reward')
            plt.savefig('results/median_rewards.png')

        if config['train'] and i % config['evaluation_frequency'] == 0 and i != 0:
            evaluation_rewards = []
            agent.eval(True)
            for ep in range(10):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                evaluation_rewards.append(episode_reward)

            median_evaluation_reward = np.median(evaluation_rewards)
            median_evaluation_rewards.append(median_evaluation_reward)
            print('Evaluation rewards: ', evaluation_rewards, ' median: ', median_evaluation_reward)
            agent.eval(False)

            plt.figure()
            plt.plot(range(len(median_evaluation_rewards)), median_evaluation_rewards)
            plt.xlabel('Episodes')
            plt.ylabel('median evaluation reward')
            plt.savefig('results/median_evaluation_rewards.png')
    
    if config['dump_frequency'] is not None:
        dc.dump_write_buffer()

    env.close()



def evaluate():
    pass

def tests():
    dqn_agent.run_tests()


if __name__ == '__main__':
    main()
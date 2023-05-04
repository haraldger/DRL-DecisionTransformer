import sys
import argparse
import time
import gym
import numpy as np
import torch
from utils import experience_replay, epsilon_scheduler, constants
from Agents import dt_agent, random_agent, dqn_agent

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
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate')
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor')
    parser.add_argument('-l', '--load', type=str, default="None", help='Load model. Provide name of model file, without extension or folder')
    
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

    if args.print_frequency:
        config['print_frequency'] = args.print_frequency

    if args.learning_rate:
        config['learning_rate'] = args.learning_rate

    if args.gamma:
        config['gamma'] = args.gamma
        

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


    # Initialize objects
    global replay_buffer
    if config['experience_replay']:
        replay_buffer = experience_replay.ReplayBuffer(config['replay_memory_size'], config['dimensions'])

    global scheduler
    if config['epsilon_scheduler']:
        scheduler = epsilon_scheduler.EpsilonScheduler(config['initial_epsilon'], config['final_epsilon'], config['decay_frames'], config['decay_mode'], config['decay_rate'])

    global agent
    if config['agent'] == 'random':
        agent = random_agent.RandomAgent(env)
    elif config['agent'] == 'dqn':
        agent = dqn_agent.DQNAgent(env, replay_buffer, scheduler, config['dqn_learning_rate'], config['gamma'])
        if config['load'] != 'None':
            agent.load(config['load'])
            save_name = config['load']      # If we are loading a model, we want to save it with the same name
    elif config['agent'] == 'dt':
        agent = dt_agent.DTAgent(env)

    if config['save'] and config['load'] == 'None':     # If we are training and not loading a model
        save_name = time.strftime("%Y%m%d-%H%M%S")
    


    # Game loop

    for i in range(config['num_episodes']):
        episode_reward = 0
        

        # Skip inactive frames
        for _ in range(inactive_frames):
            state, reward, done, info, _ = env.step(0)

        while not done:     # Run episode until done
            action = agent.act(state)
            next_state, reward, done, info, _ = env.step(action)
            episode_reward += reward
            
            if config['experience_replay']:
                replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

            if config['train']:
                agent.train()

        state, _ = env.reset()
        if config['verbose'] and i % config['print_frequency'] == 0:
            print('Episode: {}/{}. Reward: {}'.format(i, config['num_episodes'], episode_reward))

        if i % config['model_save_frequency'] == 0:
            agent.save(save_name)

    env.close()


def tests():
    dqn_agent.run_tests()


if __name__ == '__main__':
    main()
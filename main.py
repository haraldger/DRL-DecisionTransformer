import sys
import argparse
import time
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    parser.add_argument('-ie', '--initial_epsilon', type=float, help='Initial epsilon for epsilon-greedy exploration')
    parser.add_argument('-fe', '--final_epsilon', type=float, help='Final epsilon for epsilon-greedy exploration')

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

    if args.print_frequency is not None:
        config['print_frequency'] = args.print_frequency

    if args.learning_rate is not None:
        config['dqn_learning_rate'] = args.learning_rate

    if args.gamma is not None:
        config['gamma'] = args.gamma

    if args.initial_epsilon is not None:
        config['initial_epsilon'] = args.initial_epsilon

    if args.final_epsilon is not None:
        config['final_epsilon'] = args.final_epsilon
        

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
        agent = random_agent.RandomAgent(env)

    elif config['agent'] == 'dqn':
        agent = dqn_agent.DQNAgent(env, replay_buffer, scheduler, config['dqn_learning_rate'], config['gamma'])
        
        if config['load'] != 'None':
            agent.load(config['load'])
        
        if not config['train']:
            agent.eval()

    elif config['agent'] == 'dt':
        agent = dt_agent.DTAgent(env)

    


    # Game loop
    last_100_rewards = []
    mean_running_rewards = []
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

            total_frames += 1

        # Monitoring performance
        last_100_rewards.append(episode_reward)
        if len(last_100_rewards) > 100:
            last_100_rewards.pop(0)
        mean_running_rewards.append(np.mean(last_100_rewards))

        state, _ = env.reset()
        if config['verbose'] and i % config['print_frequency'] == 0:
            print('Episode: {}/{}, total iterations: {}. Mean running reward: {}'.format(i, config['num_episodes'], total_frames, np.mean(last_100_rewards)))

        if config['save'] and i % config['model_save_frequency'] == 0 and i != 0:
            save_name = time.strftime("%Y%m%d-%H%M%S")
            save_name += '_episodes_' + str(i)
            agent.save(save_name)
            # Save performance graph
            plt.plot(range(len(mean_running_rewards)), mean_running_rewards)
            plt.xlabel('Episodes')
            plt.ylabel('Mean running reward')
            plt.savefig('results/mean_rewards.png')
        

    env.close()


def tests():
    dqn_agent.run_tests()


if __name__ == '__main__':
    main()
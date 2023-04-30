import sys
import argparse
import time
import gym
from utils import experience_replay, constants
from Agents import dt_agent, random_agent, dqn_agent

config = dict()
agent = None
env = None
replay_buffer = None


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Playing MsPacman with Reinforcement Learning agents.')
    parser.add_argument('-a', '--agent', choices=['random', 'dqn', 'dt'], type=str, default='random', help='Agent to use')
    parser.add_argument('-t', '--train', action='store_true', help='Train the agent')
    parser.add_argument('-n', '--num_episodes', type=int, default=100000, help='Number of episodes to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-pf', '--print_frequency', type=int, help='Frequency in episodes to print progress')
    
    args = parser.parse_args()



    # Set configuration
    global config
    config.update(constants.load())     # Load constants

    # Set configuration based on arguments
    config['agent'] = args.agent
    config['train'] = args.train
    config['num_episodes'] = args.num_episodes
    config['verbose'] = args.verbose
    if args.print_frequency:
        config['print_frequency'] = args.print_frequency
    else:
        config['print_frequency'] = constants.PRINT_FREQUENCY

    print("Print frequency is: ", config['print_frequency'])

    if args.agent == 'random':
        config['experience_replay'] = False
    elif args.agent == 'dqn':
        config['experience_replay'] = True
    elif args.agent == 'dt':
        config['experience_replay'] = False
    else:
        print('Invalid agent')
        sys.exit()


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

    global agent
    if config['agent'] == 'random':
        agent = random_agent.RandomAgent(env)
    elif config['agent'] == 'dqn':
        agent = dqn_agent.DQNAgent(env)
    elif config['agent'] == 'dt':
        agent = dt_agent.DTAgent(env)

    global replay_buffer
    if config['experience_replay'] and config['train']:
        replay_buffer = experience_replay.ExperienceReplay()


    # Training loop

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
                replay_buffer.add(state, action, reward, done)

            state = next_state

        state, _ = env.reset()
        if config['verbose'] and i % config['print_frequency'] == 0:
            print('Episode: {}/{}. Reward: {}'.format(i, config['num_episodes'], episode_reward))

    env.close()


if __name__ == '__main__':
    main()
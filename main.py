import sys
import argparse
import time
import gym
from utils import experience_replay
from Agents import dt_agent, random_agent, dqn_agent

CONFIG = dict()
agent = None
env = None
replay_buffer = None


def main():
    parser = argparse.ArgumentParser(
        description='Playing MsPacman with Reinforcement Learning agents.')
    parser.add_argument('-a', '--agent', choices=['random', 'dqn', 'dt'], type=str, default='random', help='Agent to use')
    parser.add_argument('-t', '--train', action='store_true', help='Train the agent')
    parser.add_argument('-n', '--num_episodes', type=int, default=100000, help='Number of episodes to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    
    args = parser.parse_args()

    CONFIG['agent'] = args.agent
    CONFIG['train'] = args.train
    CONFIG['num_episodes'] = args.num_episodes
    CONFIG['verbose'] = args.verbose

    if args.agent == 'random':
        CONFIG['experience_replay'] = False
    elif args.agent == 'dqn':
        CONFIG['experience_replay'] = True
    elif args.agent == 'dt':
        CONFIG['experience_replay'] = False
    else:
        print('Invalid agent')
        sys.exit()

    global env 
    env = gym.make('ALE/MsPacman-v5')

    print('Playing MsPacman with {} agent'.format(args.agent))
    print('Mode: {}'.format('train' if args.train else 'evaluation'))

    run()


def run():
    # Reset environment
    state, _ = env.reset()
    inactive_frames = 65


    # Initialize objects

    global agent
    if CONFIG['agent'] == 'random':
        agent = random_agent.RandomAgent(env)
    elif CONFIG['agent'] == 'dqn':
        agent = dqn_agent.DQNAgent(env)
    elif CONFIG['agent'] == 'dt':
        agent = dt_agent.DTAgent(env)

    global replay_buffer
    if CONFIG['experience_replay'] and CONFIG['train']:
        replay_buffer = experience_replay.ExperienceReplay()


    # Training loop

    for i in range(CONFIG['num_episodes']):
        episode_reward = 0

        # Skip inactive frames
        for _ in range(inactive_frames):
            state, reward, done, info, _ = env.step(0)

        while not done:     # Run episode until done
            action = agent.act(state)
            next_state, reward, done, info, _ = env.step(action)
            episode_reward += reward
            
            if CONFIG['experience_replay']:
                experience_replay.add(state, action, reward, done)

            state = next_state

        state, _ = env.reset()
        if CONFIG['verbose'] and i % 1000 == 0:
            print('Episode: {}/{}. Reward: {}'.format(i, CONFIG['num_episodes'], episode_reward))

    env.close()


if __name__ == '__main__':
    main()
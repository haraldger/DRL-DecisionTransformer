import sys
import argparse
import time
import gym
from utils import experience_replay
from Agents import random_agent, dqn_agent, decision_transformer

CONFIG = dict()
agent = None
env = None


def main():
    parser = argparse.ArgumentParser(
        description='Playing MsPacman with Reinforcement Learning agents.')
    parser.add_argument('-a', '--agent', choices=['random', 'dqn', 'dt'], type=str, default='random', help='Agent to use')
    parser.add_argument('-t', '--train', action='store_true', help='Train the agent')
    parser.add_argument('-n', '--num_episodes', type=int, default=100000, help='Number of episodes to train')
    
    args = parser.parse_args()

    CONFIG['agent'] = args.agent
    CONFIG['train'] = args.train
    CONFIG['num_episodes'] = args.num_episodes

    if args.agent == 'random':
        CONFIG['experience_replay'] = False
    elif args.agent == 'dqn':
        CONFIG['experience_replay'] = True
    elif args.agent == 'dt':
        CONFIG['experience_replay'] = True
    else:
        print('Invalid agent')
        sys.exit()

    global env 
    env = gym.make('ALE/MsPacman-v5')

    print('Playing MsPacman with {} agent'.format(args.agent))
    print('Mode: {}'.format('train' if args.train else 'evaluation'))

    if args.train:
        train()
    else:
        evaluate()


def train():
    observation, _ = env.reset()
    inactive_frames = 65
    
    for i in range(inactive_frames):
        observation, _, _, _, _ = env.step(0)

    

def evaluate():
    pass


if __name__ == '__main__':
    main()
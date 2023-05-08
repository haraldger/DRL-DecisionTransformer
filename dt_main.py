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
from utils.data_transforms import image_transformation_crop_downscale_norm
from torch.utils.data import Dataset, DataLoader
from utils.data_read import DataReader

config = dict()
agent = "dt"
env = None

def run():
    # Global objects
    global config 
    global agent
    global env

    if not config['train'] and config['dump_frequency'] is not None:
        data_collector = DataCollector(config['output'], config['dump_frequency'])
        # Evaluation mode

        print("TODO: Evaluation mode not yet implemented")

    else:
        print("Save frequency: ", config['model_save_frequency_dt'])
        print("Evaluation frequency: ", config['evaluation_frequency_dt'])

        print("Loading data...")
        reader = DataReader(
            config['input_trajectory_path'], 
            transform=image_transformation_crop_downscale_norm, 
            float_state=True, 
            k_last_iters=1024,
            verbose_freq=50
        )

        print("Starting training...")
        # Training mode
        dt_model = dt_agent.DTAgent(env, config)
        dt_model.train(
            dataset=reader,
            num_epochs=config['num_epochs'],
            batch_size=1,
            verbose=config['verbose'],
            print_freq=config['print_frequency']
        )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Playing MsPacman with Reinforcement Learning agents.')
    parser.add_argument('-t', '--train', action='store_true', help='Train the agent')
    parser.add_argument('-n', '--num_epochs', type=int, default=100000, help='Number of epochs to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-pf', '--print_frequency', type=int, help='Frequency in episodes to print progress')
    parser.add_argument('-df', '--dump_frequency', type=int, default=None, help="How many episodes between writing trajectories to outfile")
    parser.add_argument('-i', '--input', type=str, default="random_trajectories.h5", help="Input trajectory path for data collection")
    parser.add_argument('-o', '--output', type=str, default="random_trajectories.h5", help="Output trajectory path for data collection")
    parser.add_argument('--evaluation_frequency', type=int, help='Frequency in episodes to evaluate model')

    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate')
    parser.add_argument('-l', '--load', type=str, default="None", help='Load model. Provide name of model file, without extension or folder')

    args = parser.parse_args()

    # Set configuration
    global config
    config.update(constants.load())     # Load constants

    # Set configuration based on arguments
    config['agent'] = 'dt'
    config['train'] = args.train
    config['input_trajectory_path'] = args.input
    config['num_epochs'] = args.num_epochs
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

    print("Print frequency is: ", config['print_frequency'])

    if config['train']:
        config['save'] = True

    # Initialize environment
    global env 
    env = gym.make('ALE/MsPacman-v5')

    print('Playing MsPacman with {} agent'.format(config['agent']))
    print('Mode: {}'.format('train' if args.train else 'evaluation'))

    # Run
    run()

if __name__ == '__main__':
    main()
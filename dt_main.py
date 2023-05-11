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
from utils.data_transforms import image_transformation_just_norm, image_transformation_grayscale_crop_downscale, image_transformation_grayscale_crop_downscale_norm
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

    dt_model = dt_agent.DTAgent(env, config)

    if not config['train'] and config['load'] is not None:
        # Evaluation mode
        
        if config['dump_frequency'] is not None:
            data_collector = DataCollector(config['output'], config['dump_frequency'])
        else:
            data_collector = None

        # Load model
        dt_model.load(config['load'])

        # Evaluate
        dt_model.model.eval()

        evaluation_rewards = []
        for eval_idx in range(config['eval_trajectories']):
            episode_reward, episode_seq_len = dt_model.run_evaluation_traj(
                data_transformation=image_transformation_grayscale_crop_downscale_norm, 
                float_state=True,
                data_collection_object=data_collector,
                debug_print_freq=config['print_frequency']
            )
            evaluation_rewards.append(episode_reward)
        mean_eval_reward = np.mean(evaluation_rewards)
        median_eval_reward = np.median(evaluation_rewards)
        standard_deviation_eval_reward = np.std(evaluation_rewards)

        print("Mean evaluation reward: ", mean_eval_reward)
        print("Median evaluation reward: ", median_eval_reward)
        print("Standard deviation evaluation reward: ", standard_deviation_eval_reward)
        
        return      
    elif config['train'] and config['load'] is not None:
        # Load and train a model
        dt_model.load(config['load'])

    # make the results folder if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Train
    print("Save frequency: ", config['model_save_frequency_dt'])
    print("Evaluation frequency: ", config['evaluation_frequency_dt'])
    print("Learning rate: ", config['learning_rate_dt'])

    # TODO: trying out first k iters rather than last k

    print("Loading data...")
    reader = DataReader(
        config['input_trajectory_path'], 
        store_transform=image_transformation_grayscale_crop_downscale, 
        store_float_state=False,
        return_transformation=image_transformation_just_norm,
        return_float_state=True, 
        k_first_iters=200,
        verbose_freq=50,
        max_ep_load=config['data_trajectories'],
        debug_print=True
    )

    print("Starting training...")
    # Training mode
    dt_model.train(
        dataset=reader,
        num_epochs=config['num_epochs'],
        batch_size=1,
        verbose=config['verbose'],
        print_freq=config['print_frequency']
    )

    return 


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Playing MsPacman with Reinforcement Learning agents.')
    parser.add_argument('-t', '--train', action='store_true', help='Train the agent')
    parser.add_argument('-et', '--eval_trajectories', type=int, default=100, help='Number of trajectories to evaluate (in evaluation mode)')
    parser.add_argument('-n', '--num_epochs', type=int, default=100000, help='Number of epochs to train')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-pf', '--print_frequency', type=int, help='Frequency in episodes to print progress')
    parser.add_argument('-df', '--dump_frequency', type=int, default=None, help="How many episodes between writing trajectories to outfile")
    parser.add_argument('-i', '--input', type=str, default="random_trajectories.h5", help="Input trajectory path for data collection")
    parser.add_argument('-o', '--eval_output', type=str, default="random_trajectories.h5", help="Output trajectory path for data collection")
    parser.add_argument('--evaluation_frequency', type=int, help='Frequency in episodes to evaluate model')

    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate')
    parser.add_argument('-l', '--load', type=str, help='Load model. Provide name of model file, with extension and folder')
    parser.add_argument('-dt', '--data_trajectories', type=int, default=10000, help='Number of trajectories loaded from file.')

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
    config['output'] = args.eval_output
    config['data_trajectories'] = args.data_trajectories
    config['eval_trajectories'] = args.eval_trajectories

    if args.print_frequency is not None:
        config['print_frequency'] = args.print_frequency

    if args.evaluation_frequency is not None:
        config['evaluation_frequency'] = args.evaluation_frequency

    if args.learning_rate is not None:
        config['learning_rate_dt'] = args.learning_rate

    print("Print frequency is: ", config['print_frequency'])

    if config['train']:
        config['save'] = True
    else:
        config['save'] = False

    # Initialize environment
    global env 
    env = gym.make('ALE/MsPacman-v5')

    print('Playing MsPacman with {} agent'.format(config['agent']))
    print('Mode: {}'.format('train' if args.train else 'evaluation'))

    # Run
    run()

if __name__ == '__main__':
    main()
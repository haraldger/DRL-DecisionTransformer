from Agents import dt_agent
import gym 
import torch
from Agents.dt_agent import DTAgent
from torch.utils.data import Dataset, DataLoader
from utils.data_read import DataReader
from utils.constants import load
from utils.data_transforms import image_transformation, image_transformation_no_norm, image_transformation_just_norm, image_transformation_grayscale_crop_downscale_norm

def test_forward_pass(config):
    env = gym.make('ALE/MsPacman-v5')

    dt_model = DTAgent(env, config)

    batch_size = 1
    seq_length = 2

    # expects state to be float normalized form
    channels = 1
    # state_seq = torch.rand(batch_size, seq_length, channels, 210, 160).float()
    state_seq = torch.rand(batch_size, seq_length, channels, 84, 84).float()
    action_seq = torch.randint(high=9, size=(batch_size, seq_length, 1))
    ret_to_go_seq = torch.tensor([10000,9998]).float().reshape(batch_size, seq_length, 1)
    timestep_seq = torch.tensor([0,1]).reshape(batch_size, seq_length, 1)

    next_action = dt_model.predict_next_action(state_seq, action_seq, ret_to_go_seq, timestep_seq)
    print("Next action prediction test: ", next_action)


def test_traj(config):
    env = gym.make('ALE/MsPacman-v5')
    dt_model = DTAgent(env, config)

    reward, seq_length = dt_model.run_evaluation_traj(data_transformation=image_transformation_grayscale_crop_downscale_norm, float_state=True)
    
    print("reward: ", reward)
    print("seq_length: ", seq_length)


def test_train(config):
    # Assumes that you have run the data collection test
    # And the file "test_traj_long.h5" exists

    env = gym.make('ALE/MsPacman-v5')
    dt_model = DTAgent(env, config)

    # reader = DataReader("test_traj_long.h5", transform=image_transformation, float_state=True, k_last_iters=1024)
    reader = DataReader("test_traj_long.h5", transform=image_transformation_grayscale_crop_downscale_norm, float_state=True, k_last_iters=1024)
    # reader = DataReader("test_traj_long.h5", transform=image_transformation_just_norm, float_state=True, k_last_iters=32)

    dt_model.train(
        dataset=reader,
        num_epochs=1,
        batch_size=1,
        print_freq=1
    )


def run():
    config = load()
    print("Testing DT Agent forward pass.\n")
    test_forward_pass(config)
    print("\n")
    print("Testing DT Agent training.\n")
    test_train(config)
    print("\n")
    print("Testing a DT trajectory")
    test_traj(config)
    print("\n")
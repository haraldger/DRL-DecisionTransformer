from Agents import dt_agent
import gym 
import torch
from Agents.dt_agent import DTAgent
from torch.utils.data import Dataset, DataLoader
from utils.data_read import DataReader
from utils.constants import load
from utils.data_transforms import image_transformation_grayscale_crop_downscale_v2, image_transformation_just_norm, image_transformation_just_norm, image_transformation_grayscale_crop_downscale_norm_v2
import sys

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

    reward, seq_length = dt_model.run_evaluation_traj(data_transformation=image_transformation_grayscale_crop_downscale_norm_v2, float_state=True, debug_print_freq=50)

    print("reward: ", reward)
    print("seq_length: ", seq_length)


def test_train(config):
    # Assumes that you have run the data collection test
    # And the file "test_traj_long.h5" exists

    env = gym.make('ALE/MsPacman-v5')
    dt_model = DTAgent(env, config)

    config['save'] = True
    config['model_save_frequency_dt'] = 2
    config['evaluation_frequency_dt'] = 2

    # reader = DataReader("test_traj_long.h5", transform=image_transformation_just_norm, float_state=True, k_last_iters=32)
    reader = DataReader(
        "test_traj_long.h5", 
        store_transform=image_transformation_grayscale_crop_downscale_v2, 
        store_float_state=False,
        return_transformation=image_transformation_just_norm,
        return_float_state=True,
        k_last_iters=200)

    dt_model.train(
        dataset=reader,
        num_epochs=1,
        batch_size=1,
        verbose=True,
        print_freq=1
    )

def debug_loss(config):
    # Test that the loss function works

    env = gym.make('ALE/MsPacman-v5')
    dt_model = DTAgent(env, config)

    # fake output
    # two batches of seq_length 2
    output = torch.zeros(2,2,9).float().to(dt_model.device)
    output[0,0,2] = 1.0
    output[0,1,3] = 1.0
    output[1,0,4] = 1.0
    output[1,1,5] = 1.0

    # fake target
    labels = torch.zeros(2,2,1).long().to(dt_model.device)
    labels[0,0] = 2
    labels[0,1] = 3
    labels[1,0] = 4
    labels[1,1] = 5

    loss = dt_model.cross_entropy_loss(output.reshape(-1, 9), labels.reshape(-1))

    if loss != 0:
        print("Loss function test failed  Should have been 0. Loss: ", loss)
        sys.exit()

    output[0,0,2] = 0.5
    output[0,0,3] = 0.5

    loss = dt_model.cross_entropy_loss(output.reshape(-1, 9), labels.reshape(-1))

    if loss == 0:
        print("Loss function test failed  Should have been non-zero. Loss: ", loss)
        sys.exit()
    else:
        print("Loss function test passed. Loss: ", loss)


def run():
    config = load()
    config['save'] = False
    print("Testing DT Agent loss function.\n")
    debug_loss(config)
    print("\n")
    print("Testing DT Agent forward pass.\n")
    print("\n")
    print("Testing a DT trajectory")
    test_traj(config)
    print("\n")
    test_forward_pass(config)
    print("\n")
    print("Testing DT Agent training.\n")
    test_train(config)
    
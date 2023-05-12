import sys
import numpy as np
import random
import h5py
import torch
import cv2 as cv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_transforms import image_transformation, image_transformation_no_norm, image_transformation_crop_downscale, image_transformation_just_norm, image_transformation_grayscale_crop_downscale_norm, image_transformation_grayscale_crop_downscale
from utils.data_read import DataReader
import collections

if __name__ == "__main__":
    # DATA_FILE = "dqn_dataset.h5"
    DATA_FILE = "random_trajectories.h5"

    # Read in the data
    reader = DataReader(
        DATA_FILE, 
        store_transform=image_transformation_grayscale_crop_downscale, 
        store_float_state=False, 
        return_transformation=image_transformation_just_norm,
        return_float_state=True,
        k_first_iters=200,
        verbose_freq=50,
        max_ep_load=100,
        debug_print=True
    )

    print("Number of data trajectories: ", len(reader.all_traj_data))

    dataloader = DataLoader(reader, batch_size=30, shuffle=True)

    for batch_idx, (states, actions, rewards, rewards_to_go, timesteps, dones) in enumerate(dataloader):
        print(actions.shape)
        print(actions[0])

        sys.exit("Only one batch")
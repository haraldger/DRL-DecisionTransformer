import sys
import numpy as np
import random
import h5py
import torch
import cv2 as cv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from PIL import Image

image_transformation = torch.nn.Sequential(
    transforms.Grayscale(),
    transforms.Normalize(0, 255)
)

image_transformation_no_norm = torch.nn.Sequential(
    transforms.Grayscale(),
)

image_transformation_just_norm = torch.nn.Sequential(
    transforms.Normalize(0, 255)
)

image_transformation_grayscale_crop_downscale_norm = torch.nn.Sequential(
    # Crop bottom 34 pixels and resize to 84x84
    transforms.Grayscale(),
    transforms.CenterCrop((210, 160 - 38)),
    transforms.Resize((84,84), interpolation=Image.BICUBIC),
    transforms.Normalize(0,255)
)

image_transformation_grayscale_crop_downscale = torch.nn.Sequential(
    # Crop bottom 34 pixels and resize to 84x84
    transforms.Grayscale(),
    transforms.CenterCrop((210, 160 - 38)),
    transforms.Resize((84,84), interpolation=Image.BICUBIC)
)

image_transformation_crop_downscale = torch.nn.Sequential(
    transforms.CenterCrop((210, 160 - 38)),
    transforms.Resize((84,84), interpolation=Image.BICUBIC),
)

class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)

crop_bottom = LambdaModule(lambda x: x[:, :, :-38, :])

image_transformation_grayscale_crop_downscale_norm_v2 = torch.nn.Sequential(
    # Crop bottom 34 pixels and resize to 84x84
    transforms.Grayscale(),
    crop_bottom,
    transforms.Resize((84,84), interpolation=Image.BICUBIC),
    transforms.Normalize(0,255)
)

image_transformation_grayscale_crop_downscale_v2 = torch.nn.Sequential(
    # Crop bottom 34 pixels and resize to 84x84
    transforms.Grayscale(),
    crop_bottom,
    transforms.Resize((84,84), interpolation=Image.BICUBIC)
)

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
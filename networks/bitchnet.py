import numpy as np
import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.bn1(out) 
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) 
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) 

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bitchnet(nn.Module):
    def __init__(self, in_channels, output_dim=768) -> None:
        super().__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        # Batch normalization after the initial convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        # Max pooling after the batch normalization layer
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # ResNet layers
        self.layer1 = Bottleneck(64, 64)
        self.layer2 = Bottleneck(64, 128)
        self.layer3 = Bottleneck(128, 256)
        self.layer4 = Bottleneck(256, 512)

        # Linear output layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(15360, output_dim)

    def forward(self, x):
        # Initial convolutional layer
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.maxpool_3(c1)

        # ResNet layers
        c2 = self.layer1(c1)
        c2 = self.maxpool_2(c2)

        c3 = self.layer2(c2)
        c3 = self.maxpool_2(c3)

        c4 = self.layer3(c3)
        c4 = self.maxpool_2(c4)

        c5 = self.layer4(c4)

        # Linear output layer
        out = self.flatten(c5)
        out = self.linear(out)

        # Return learned encoding
        return out
    

# Tests

def test_bitchnet_forward():
    x = torch.randn(64, 3, 210, 160)    # Batch size 64, 3 channels, 210x160 pixels
    model = Bitchnet(3)
    y = model(x)
    assert y.shape == (64, 768), f"Bad shape: {y.shape}"

    print("Test passed!")

def test_large_bitchnet_forward():
    x = torch.randn(1000, 3, 210, 160)    # Batch size 64, 3 channels, 210x160 pixels
    model = Bitchnet(3)
    y = model(x)
    assert y.shape == (1000, 768), f"Bad shape: {y.shape}"

    print("Test passed!")

def run_tests():
    test_bitchnet_forward()
    test_large_bitchnet_forward()
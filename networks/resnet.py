import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
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
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
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
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) 
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) 

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block_type, layers, in_channels=3, output_dim=384) -> None:
        super().__init__()
        self.block_type = block_type
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        # Batch normalization after the initial convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        # Max pooling after the batch normalization layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self.make_layer(64, 64, layers[0])
        self.layer2 = self.make_layer(64, 128, layers[1])
        self.layer3 = self.make_layer(128, 256, layers[2])
        self.layer4 = self.make_layer(256, 512, layers[3])

        # Linear output layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(17920, output_dim)


        
    def forward(self, x):
        # Initial convolutional layer
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.maxpool(c1)

        # ResNet layers
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Linear output layer
        out = self.flatten(c5)
        out = self.linear(out)

        # Return learned encoding
        return out

        
    
    def make_layer(self, in_channels, out_channels, num_blocks):
        blocks = []

        # Create the first block separately to perform downsampling
        if in_channels != out_channels:
            blocks.append(self.block_type(in_channels, out_channels, stride=2))
        else:
            blocks.append(self.block_type(in_channels, out_channels, stride=1))

        # Create the rest of the blocks
        for i in range(1, num_blocks):    
            blocks.append(self.block_type(out_channels, out_channels, stride=1))

        return nn.Sequential(*blocks)

    
def resnet18(in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels)

def resnet34(in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels)

def resnet50(in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels)

def resnet101(in_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels)

def resnet152(in_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels)


# Test the network

def test_network_and_output_shapes():
    print('Testing network and output shapes...')

    net = resnet18()
    x = torch.randn(2, 3, 210, 160)
    print(f'Input shape = {x.shape}')
    print(f'Batch size = {x.shape[0]}')
    y = net(x)
    print(f'ResNet18 output shape = {y.shape}')
    assert y.shape == (2, 384)

    net = resnet34()
    y = net(x)
    print(f'ResNet34 output shape = {y.shape}')
    assert y.shape == (2, 384)

    net = resnet50()
    y = net(x)
    print(f'ResNet50 output shape = {y.shape}')
    assert y.shape == (2, 384)

    net = resnet101()
    y = net(x)
    print(f'ResNet101 output shape = {y.shape}')
    assert y.shape == (2, 384)

    net = resnet152()
    y = net(x)
    print(f'ResNet152 output shape = {y.shape}')   
    assert y.shape == (2, 384)

    print('Tests passed!')

def run_tests():
    test_network_and_output_shapes()


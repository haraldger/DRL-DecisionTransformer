import numpy as np
import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
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
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
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
    
class ResNet(nn.Module):
    def __init__(self, block_type, layers) -> None:
        super().__init__()
        self.block_type = block_type
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Batch normalization after the initial convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        # Max pooling after the batch normalization layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self.make_layer(64, 64, layers[0])
        self.layer2 = self.make_layer(64, 128, layers[1])
        self.layer3 = self.make_layer(128, 256, layers[2])
        self.layer4 = self.make_layer(256, 512, layers[3])


        
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

        # Return learned features at different scales
        return [c1, c2, c3, c4, c5]

        
    
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

    
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# Test the network

def test_network_and_output_shapes():
    net = resnet18()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(f'ResNet18 network = {net}')
    print(f'ResNet18 output shape = {y[-1].shape}')

    net = resnet34()
    y = net(x)
    print(f'ResNet34 network = {net}')
    print(f'ResNet34 output shape = {y[-1].shape}')

    net = resnet50()
    y = net(x)
    print(f'ResNet50 network = {net}')
    print(f'ResNet50 output shape = {y[-1].shape}')

    net = resnet101()
    y = net(x)
    print(f'ResNet101 network = {net}')
    print(f'ResNet101 output shape = {y[-1].shape}')

    net = resnet152()
    y = net(x)
    print(f'ResNet152 network = {net}')
    print(f'ResNet152 output shape = {y[-1].shape}')

def test_intermediate_output_shapes():
    net = resnet18()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(f'ResNet18 C1 output shape = {y[0].shape}') 
    print(f'ResNet18 C2 output shape = {y[1].shape}')
    print(f'ResNet18 C3 output shape = {y[2].shape}')
    print(f'ResNet18 C4 output shape = {y[3].shape}')
    print(f'ResNet18 C5 output shape = {y[4].shape}')

    net = resnet34()
    y = net(x)
    print(f'ResNet34 C1 output shape = {y[0].shape}')
    print(f'ResNet34 C2 output shape = {y[1].shape}')
    print(f'ResNet34 C3 output shape = {y[2].shape}')
    print(f'ResNet34 C4 output shape = {y[3].shape}')
    print(f'ResNet34 C5 output shape = {y[4].shape}')

    net = resnet50()
    y = net(x)
    print(f'ResNet50 C1 output shape = {y[0].shape}')
    print(f'ResNet50 C2 output shape = {y[1].shape}')
    print(f'ResNet50 C3 output shape = {y[2].shape}')
    print(f'ResNet50 C4 output shape = {y[3].shape}')
    print(f'ResNet50 C5 output shape = {y[4].shape}')

    net = resnet101()
    y = net(x)
    print(f'ResNet101 C1 output shape = {y[0].shape}')
    print(f'ResNet101 C2 output shape = {y[1].shape}')
    print(f'ResNet101 C3 output shape = {y[2].shape}')
    print(f'ResNet101 C4 output shape = {y[3].shape}')
    print(f'ResNet101 C5 output shape = {y[4].shape}')

    net = resnet152()
    y = net(x)
    print(f'ResNet152 C1 output shape = {y[0].shape}')
    print(f'ResNet152 C2 output shape = {y[1].shape}')
    print(f'ResNet152 C3 output shape = {y[2].shape}')
    print(f'ResNet152 C4 output shape = {y[3].shape}')
    print(f'ResNet152 C5 output shape = {y[4].shape}')


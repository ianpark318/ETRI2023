from netrc import netrc

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

import torchsummary

class conv3x3(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=nChannels, out_channels=nOutChannels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

class conv1x1(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=nChannels, out_channels=nOutChannels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

class Denseblock(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super().__init__()

        self.conv1 = conv3x3(nChannels=nChannels, nOutChannels=nOutChannels)
        self.conv2 = conv3x3(nChannels=nOutChannels, nOutChannels=nOutChannels)
        self.conv3 = conv3x3(nChannels=nOutChannels, nOutChannels=nOutChannels)
        self.conv4 = conv1x1(nChannels=nOutChannels, nOutChannels=nOutChannels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

## growthRate = 12
## 224 -
class DenseNet(nn.Module):
    def __init__(self, nClasses):
        super().__init__()

        self.dense1 = Denseblock(nChannels=1, nOutChannels=64)
        self.dense2 = Denseblock(nChannels=64, nOutChannels=64)
        self.dense3 = Denseblock(nChannels=64, nOutChannels=64)
        self.dense4 = Denseblock(nChannels=64, nOutChannels=64)
        # self.dense5 = Denseblock(in_channels=3, out_channels=64)

        self.fc = nn.Linear(64, nClasses)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv128 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv192 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.dense1(x)
        x2 = self.dense2(x1)

        cc = torch.concat((x2, x1), 1)
        x3 = self.conv128(cc)
        x3 = self.dense3(x3)

        cc = torch.concat((x3, x2, x1), 1)
        x4 = self.conv192(cc)
        x4 = self.dense4(x4)

        out = self.gap(x4)
        out = torch.squeeze(out)
        out = F.log_softmax(self.fc(out))

        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = DenseNet(nClasses=47).to(device)

#torchsummary.summary(net, (3, 224, 224))
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

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super().__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(in_channels=nChannels, out_channels=interChannels, kernel_size=1, bias=False)

        nChannels = interChannels
        self.bn2 = nn.BatchNorm2d(nChannels)
        self.conv2 = nn.Conv2d(in_channels=nChannels, out_channels=growthRate, kernel_size=3, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = torch.cat((x, out), 1)

        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = torch.cat((x, out), 1)

        return out



class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=nChannels, out_channels=nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = F.avg_pool2d(x, 2)

        return x

'''
Growth Rate: concat 을 위해서 각 layer의 output이 똑같은 channel로 해주는 것이 좋은데, 이 channel의 수를 Growth Rate

Composite Function: H_l() 으로 표현되고, BN-ReLU-Conv3x3 으로 이뤄짐

Bottlenect layers: 1x1 conv 를 사용해서 channel을 줄이고, 이후에 보통 학습을 위한 3x3 conv를 사용해서 weight를 줄임,
                   BN-ReLU-1x1conv-BN-ReLU-3x3conv 로 이뤄짐

Transition layers (Pooling layer): block이 끝난 후에 pooling layer를 사용해서 feature의 weight를 줄임

Compression: pooling layer의 1x1conv에서 channel을 줄여주는 비율
'''

'''
DenseBlock을 이루는 layer의 인풋은 k_0 + k * (l - 1) 만큼 늘어남 이후, forward 에서 layer를 지날 때 마다 concat하여 다음 layer 통과

'''


class DensNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super().__init__()

        def DenseBlock(self, nChannels, growthRate, nDenseBlocks, bottleneck):
            layers = []

            for i in range(int(nDenseBlocks)):
                if bottleneck:
                    layers.append(Bottleneck(nChannels, growthRate))
                else:
                    layers.append(SingleLayer(nChannels, growthRate))

                nChannels += growthRate

            return nn.Sequential(*layers)



        nDenseBlocks = (depth-4) // 3  ## the number of DenseBlock

        if bottleneck:
            nDenseBlocks //= 2


        ## before Dense
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nChannels, kernel_size=3, padding=1, bias=False)

        ## 1st Dense
        self.dense1 = DenseBlock(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        ## 2nd Dense
        nChannels = nOutChannels
        self.dense2 = DenseBlock(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        ## 3rd Dense
        nChannels = nOutChannels
        self.dense3 = DenseBlock(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        ## pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels, nClasses)
        self.nChannels = nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trnas2(self.dense2(x))
        x = self.dense3(x)

        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = torch.squeeze(x)
        x = F.lob_softmax(self.fc(x))

        return x
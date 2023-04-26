import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super().__init__()
        interChannels = 4 * growthRate
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

        nDenseBlocks = (depth - 4) // 3  ## the number of DenseBlock

        if bottleneck:
            nDenseBlocks //= 2

        ## before Dense
        nChannels = 2 * growthRate
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
        x = F.log_softmax(self.fc(x))

        return x


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        # self.output_dim = output_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.flatten = nn.Flatten()
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        bias=False,
        padding=1,
        padding_mode='zeros'
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        padding=1,
        padding_mode='zeros'
    )


class IdentityBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(IdentityBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.conv2 = conv3x3(out_planes, out_planes, 1)

        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, in_planes, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.in_planes = in_planes

        self.conv = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self.make_layer(block, 64, num_blocks[3], stride=1)
        self.layer5 = self.make_layer(block, 64, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # 64 112 112
        out = self.avgpool(out)  # 64 1 1
        out = torch.flatten(out, 1)
        return out


def ResNet18(in_planes, num_classes):
    return ResNet(block=IdentityBlock, in_planes=in_planes, num_blocks=[2, 2, 2, 2], num_classes=num_classes)


class D2GMNet(nn.Module):
    def __init__(self):
        super(D2GMNet, self).__init__()
        self.resnet18 = ResNet18(64, 15)
        self.lstm = LSTM(input_dim=2, hidden_dim=64, seq_len=240, layers=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15424, 8192)
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 15)

    def forward(self, acc_x, et_x):
        acc_out = self.resnet18(acc_x)
        et_out = self.lstm(et_x)
        # print(acc_out.shape, et_out.shape)
        output = torch.cat((acc_out, et_out), dim=1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        output = F.softmax(output, dim=1)
        return output

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
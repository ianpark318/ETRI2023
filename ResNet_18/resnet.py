import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        bias=False,
        padding = 1,
        padding_mode='zeros'
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        padding = 1,
        padding_mode='zeros'
    )


class IdentityBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1):
        super(IdentityBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.conv2 = conv3x3(out_planes, out_planes, 1)

        self.bn1   = nn.BatchNorm2d(out_planes)
        self.bn2   = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        identity = x
        print('input shape:', x.shape)
        out  = self.conv1(x)
        print('output shape:', out.shape)
        out  = self.bn1(out)
        print('output shape:', out.shape)
        out  = F.relu(out)
        print('output shape:', out.shape)
        out  = self.conv2(out)
        print('output shape:', out.shape)
        out  = self.bn2(out)
        out += identity
        print('output shape:', out.shape)
        out  = F.relu(out)
        print('output shape:', out.shape)
        return out


class ResNet(nn.Module):
    def __init__(self, block, in_planes, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.in_planes = in_planes

        self.conv = nn.Conv2d(3, self.in_planes, kernel_size = 3, stride = 1, padding = 1, padding_mode='zeros', bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self.make_layer(block, 64, num_blocks[3], stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(64, num_classes)

    def make_layer(self, block, out_planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks -1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, out_planes))
                self.in_planes = out_planes
            return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(in_planes, num_classes):
    return ResNet(block = IdentityBlock, in_planes = in_planes, num_blocks = [2, 2, 2, 2], num_classes = num_classes)
import torch

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import argparse

import os

import shutil

import setproctitle

import model
# import make_graph

parser = argparse.ArgumentParser(description="Train the DenseNet")

parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('--layers', default=100, type=int)
parser.add_argument('no-cuda', action='store_true')
parser.add_argument('--save')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--opt', default='sgd', type=str,
                    choices=('sgd', 'adam', 'rmsprop'))

# number of new channels per layer
parser.add_argument('--growth', default=12, type=int)
# dropout probability
parser.add_argument('--droprate', default=0, type=float)
# whether to use standard augmentation
parser.add_argument('--no-augment', dest='augment', action='store_false')
# compression rate in transition state
parser.add_argument('--reduce', default=0.5, type=float)
# to not uset bottlenect block
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false')
# path to latest checkpoint
parser.add_argument('--resume', default='', type=str)
# name of experiment
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str)
# Log progress to TensorBoard
parser.add_argument('--tensorboard', action='store_true')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0


def main():
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # normMean = [0.49139968, 0.48215827, 0.44653124]
    # normStd = [0.24703233, 0.24348505, 0.26158768]
    # normTransform = transforms.Normalize(normMean, normStd)

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])


    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    trainLoader = DataLoader()
    testLoader = DataLoader()

    net = model.DensNet(growthRate=args.growth, depth=100, reduction=args.reduce,
                        bottleneck=args.bottleneck, nClasses=10)

    print(f' + Number of params: {sum([p.data.nelement() for p in net.parameters()])}')

    if args.cuda:
        net = net.cuda()

    # if args.opt == 'sgd':

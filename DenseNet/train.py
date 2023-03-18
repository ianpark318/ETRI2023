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
parser.add_argument('--opt', default='adam', type=str,
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

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                              momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()


def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100. * incorrect / len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader)  # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100. * incorrect / nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    main()
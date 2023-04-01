import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from densenet import DenseNet
import argparse
from dataset import MFCCdataset, RPdataset
import numpy as np

import os

import shutil

# import setproctitle


def main():
    num_epoch = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_file_mfcc = '../data.csv'
    csv_file_rp = '../rpdata.csv'
    model = DenseNet(nClasses=47).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = MFCCdataset(csv_file_mfcc)
    #dataset = RPdataset(csv_file_rp)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - validation_size
    transform = transforms.Compose([transforms.Resize((20, 4)),
                                    transforms.ToTensor()])

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

    # validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(num_epoch):
        print('EPOCH {}:'.format(epoch + 1))
        training_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {training_loss / 100:.3f}')
                training_loss = 0.0

            if i % 1000 == 999:
                # test with validation set if val_loader exist
                with torch.no_grad():
                    val_loss = []
                    for j, val_data in enumerate(validation_dataloader):
                        val_inputs, val_labels = val_data
                        val_inputs = val_inputs.to(device)
                        val_labels = val_labels.to(device)
                        val_outputs = model(val_inputs)
                        val_loss.append(criterion(val_outputs, val_labels).item())
                    print("validation loss {}".format(np.mean(val_loss)))

    print('Finished Training')
    PATH = './MFCC_densenet.pth'
    #PATH = './RP_densenet.pth'
    torch.save(model.state_dict(), PATH)

    # test
    with torch.no_grad():
        test_loss = []
        for k, test_data in enumerate(test_dataloader):
            test_inputs, test_labels = test_data
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_inputs)
            test_loss.append(criterion(test_outputs, test_labels).item())
        print("test loss {}".format(np.mean(test_loss)))


if __name__ == '__main__':
    main()
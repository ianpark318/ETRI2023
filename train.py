import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import argparse
from dataset import RPdataset, ETdataset, RPnETdataset
from model import D2GMNet
import numpy as np
import tqdm
import os

import shutil

# import setproctitle


def main():
    num_epoch = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_file_train = '/home/jh/ETRIdata/ETRItrain.csv'
    csv_file_val = '/home/jh/ETRIdata/ETRIval.csv'
    csv_file_test = '/home/jh/ETRIdata/ETRItest.csv'
    model = D2GMNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 800], gamma=0.5)

    train_dataset1 = RPdataset(csv_file_train)
    train_dataset2 = ETdataset(csv_file_train)
    val_dataset1 = RPdataset(csv_file_val)
    val_dataset2 = ETdataset(csv_file_val)
    test_dataset1 = RPdataset(csv_file_test)
    test_dataset2 = ETdataset(csv_file_test)
    dataset_size = len(train_dataset1)
    # train_size = int(dataset_size * 0.8)
    # validation_size = int(dataset_size * 0.1)
    # test_size = dataset_size - train_size - validation_size
    # transform = transforms.Compose([transforms.Resize((20, 4)),
    #                                 transforms.ToTensor()])

    # train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # print(f"Training Data Size : {len(train_dataset1)}")
    # print(f"Validation Data Size : {len(validation_dataset)}")
    # print(f"Testing Data Size : {len(test_dataset)}")
    train_dataloader1 = DataLoader(train_dataset1, batch_size=16, shuffle=True, drop_last=True)
    train_dataloader2 = DataLoader(train_dataset2, batch_size=16, shuffle=True, drop_last=True)
    val_dataloader1 = DataLoader(val_dataset1, batch_size=4, shuffle=True, drop_last=True)
    val_dataloader2 = DataLoader(val_dataset2, batch_size=4, shuffle=True, drop_last=True)
    test_dataloader1 = DataLoader(test_dataset1, batch_size=4, shuffle=True, drop_last=True)
    test_dataloader2 = DataLoader(test_dataset2, batch_size=4, shuffle=True, drop_last=True)

    # validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(num_epoch):
        print('EPOCH {}:'.format(epoch + 1))
        training_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(zip(train_dataloader1, train_dataloader2))):
            # get the inputs; data is a list of [inputs, labels]
            data1, data2 = data
            rp, labels = data1
            et, _ = data2
            # rp, et = inputs[0], inputs[1]
            rp = rp.to(device)
            et = et.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(rp, et)
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
                    for j, val_data in enumerate(zip(val_dataloader1, val_dataloader2)):
                        val_data1, val_data2 = val_data
                        val_rp, val_labels = val_data1
                        val_et, _ = val_data2
                        # val_rp, val_et = val_inputs[0], val_inputs[1]
                        val_rp = val_rp.to(device)
                        val_et = val_et.to(device)
                        val_labels = val_labels.to(device)
                        val_outputs = model(val_rp, val_et)
                        val_loss.append(criterion(val_outputs, val_labels).item())
                    print("validation loss {}".format(np.mean(val_loss)))
        scheduler.step()

    print('Finished Training')
    PATH = 'DenseNet/MFCC_densenet.pth'
    #PATH = './RP_densenet.pth'
    torch.save(model.state_dict(), PATH)

    # test
    # with torch.no_grad():
    #     test_loss = []
    #     for k, test_data in enumerate(test_dataloader):
    #         test_rp, test_et, test_labels = test_data
    #         # test_rp, test_et = test_inputs[0], test_inputs[1]
    #         test_rp = test_rp.to(device)
    #         test_et = test_et.to(device)
    #         test_labels = test_labels.to(device)
    #         test_outputs = model(test_rp, test_et)
    #         test_loss.append(criterion(test_outputs, test_labels).item())
    #     print("test loss {}".format(np.mean(test_loss)))


if __name__ == '__main__':
    main()
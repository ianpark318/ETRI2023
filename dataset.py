import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import cv2

class RPdataset(Dataset):
    def __init__(self, csv_name, transform=None):
        super(RPdataset, self).__init__()
        df = pd.read_csv(csv_name)

        le = LabelEncoder()
        le = le.fit(df['action'])
        df['action'] = le.transform(df['action'])

        self.x_file = list(df['img_path'])
        self.y_data = list(df['action'])
        self.transform = transforms.Resize((224, 224))

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_file)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        file = self.x_file[idx]
        img = Image.open(file)
        arr = np.asarray(img.convert('RGB'))
        arr = torch.FloatTensor(arr).permute(2, 0, 1).contiguous()
        arr = self.transform(arr)
        label = self.y_data[idx]

        return arr, label


class ETdataset(Dataset):
    def __init__(self, csvfile, transform=None):
        super(ETdataset, self).__init__()
        df = pd.read_csv(csvfile)
        self.x_file = list(df['eda_temp'])
        self.y_data = list(df['action'])

  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_file)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        file = self.x_file[idx]

        df2 = pd.read_csv(file)
        df2 = df2.fillna(0)
        eda = np.array(list(df2['eda']))
        temp = np.array(list(df2['temp']))
        et = np.column_stack([eda, temp])
        arr = torch.FloatTensor(et)
        label = self.y_data[idx]

        return arr, label

class RPnETdataset(Dataset):
    def __init__(self, csv_name, transform=None):
        super(RPnETdataset, self).__init__()
        df = pd.read_csv(csv_name)

        le = LabelEncoder()
        le = le.fit(df['action'])
        df['action'] = le.transform(df['action'])

        self.x_file1 = list(df['img_path'])
        self.transform = transforms.Resize((224, 224))
        self.x_file2 = list(df['eda_temp'])
        self.y_data = list(df['action'])
        self.transform = transforms.Resize((224, 224))

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_file2)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        file = self.x_file1[idx]
        img = Image.open(file)
        arr1 = np.asarray(img.convert('RGB'))
        arr1 = torch.FloatTensor(arr1).permute(2, 0, 1).contiguous()
        arr1 = self.transform(arr1)
        label = torch.FloatTensor(self.y_data[idx])

        df2 = pd.read_csv(self.x_file2[idx])
        df2 = df2.fillna(0)
        eda = np.array(list(df2['eda']))
        temp = np.array(list(df2['temp']))
        et = np.column_stack([eda, temp])
        arr2 = torch.FloatTensor(et)
        print(arr1.shape, arr2.shape)
        return arr1, arr2, label
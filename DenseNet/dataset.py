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

class MFCCdataset(Dataset):
    def __init__(self, csv_name, transform=None):
        super(MFCCdataset, self).__init__()
        df = pd.read_csv(csv_name)
        self.x_file = list(df.iloc[:, 0])
        self.y_data = list(df.iloc[:, 2])


  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_file)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        file = self.x_file[idx]
        img = Image.open('../' + file)
        arr = np.asarray(img)
        arr = torch.FloatTensor(arr)
        arr = arr.view([-1, 20, 4])
        label = self.y_data[idx]

        return arr, label


class RPdataset(Dataset):
    def __init__(self, csv_name, transform=None):
        super(RPdataset, self).__init__()
        df = pd.read_csv(csv_name)
        self.x_file = list(df.iloc[:, 0])
        self.y_data = list(df.iloc[:, 2])
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
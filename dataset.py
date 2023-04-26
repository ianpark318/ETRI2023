from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class RPDataset(Dataset):
    def __init__(self, df, rp_path_list, label_list, tfms=None):
        super().__init__()
        self.df = df
        self.rp_path_list = rp_path_list
        self.label_list = label_list
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.rp_path_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.tfms(image=img)['image']
        image = torch.tensor(np.array(image)).permute(2, 0, 1)

        if self.label_list is not None:
            label = self.label_list[idx]
            return image, label
        else:
            return image


class GpsDataset(Dataset):
    def __init__(self, df, lat_path_list, lon_path_list, label_list, tfms=None):
        super(GpsDataset, self).__init__()
        self.df = df
        self.lat_path_list = lat_path_list
        self.lon_path_list = lon_path_list
        self.label_list = label_list
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        lat = self.lat_path_list[idx]
        lon = self.lon_path_list[idx]
        feature_map = torch.tensor(np.array([lat, lon]))

        if self.label_list is not None:
            label = self.label_list[idx]
            return feature_map, label
        else:
            return feature_map
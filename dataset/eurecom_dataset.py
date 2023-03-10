import os
import sys
import numpy
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

sys.path.append('../')
data_root = os.path.expanduser('~//yq_pointnet//EURECOM_Kinect_Face_Dataset')

def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

class Eurecom_Dataset(Dataset):
    """
    Args:

    """

    def __init__(self, csv_path, transform=None):
        csv_path = os.path.expanduser(csv_path)
        assert os.path.exists(csv_path), '%s not found' % csv_path

        # read csv file
        self.df = pd.read_csv(csv_path, header=0, names=['point_cloud_path', 'cls_name'])
        self.df.cls_name, _ = pd.factorize(self.df.cls_name, sort=True)

        self._num_of_classes = len(self.df.cls_name.drop_duplicates())
        self._len_of_dataset = len(self.df.cls_name)
        self._transform = transform

    def get_num_of_classes(self):
        return self._num_of_classes

    def __len__(self):
        return self._len_of_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        point_cloud_path, cls_id = self.df.iloc[idx]
        if int(cls_id) == 52:
            point_cloud_data = np.loadtxt(point_cloud_path, delimiter=',')
            point_cloud_data = rand_row(point_cloud_data, 4000)
            point_cloud_data = point_cloud_data[:, 0:3]
        else:
            path = point_cloud_path
            point_cloud_data = []
            with open(path) as file:
                while 1:
                    line = file.readline()
                    if not line:
                        break
                    paras = line.split(" ")
                    if paras[0] == "v" and abs(int(paras[3])) < 1000 and int(paras[3]) != 0:
                        point_cloud_data.append((float(paras[1]), float(paras[2]), float(paras[3][0:-1])))
            random.shuffle(point_cloud_data)
            if len(point_cloud_data) < 10000:
                for i in range(10000):
                    point_cloud_data.append((0, 0, 0))
            point_cloud_data = point_cloud_data[0:4000]

        if numpy.any(numpy.isnan(point_cloud_data)):
            point_cloud_data[np.isnan(point_cloud_data)] = 0

        point_cloud_data = point_cloud_data - np.expand_dims(np.mean(point_cloud_data, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_cloud_data ** 2, axis=1)), 0)
        point_cloud_data = point_cloud_data / dist  # scale
        point_cloud_data = torch.from_numpy(point_cloud_data.astype(np.float))
        cls_id = torch.from_numpy(np.array([cls_id]).astype(np.int64))
        return point_cloud_data, cls_id

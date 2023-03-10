import os
import open3d as o3d
import numpy
import numpy as np
import pandas as pd
import torch
from numpy import zeros
from torch.utils.data import Dataset, DataLoader
from utils.readbnt import read_bntfile

data_root = os.path.expanduser('~//yq_pointnet//attack//CW//AdvData//PointNet')

np.set_printoptions(suppress=True)
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)
numpy.set_printoptions(precision=4)



def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

def get_mean_var(input_data_path):
    ipt_ori = np.loadtxt(input_data_path, delimiter=',')
    #ipt, mu, st = normalization(ipt)
    # ipt_ori = rand_row(ipt_ori, 4000)
    ipt = ipt_ori[:, 0:3]
    ipt_c4c5 = ipt_ori[:, 3:5]
    #ipt = np.append(ipt, ipt, axis=0)
    mean = np.expand_dims(np.mean(ipt, axis=0), 0)
    ipt = ipt - mean
    var = np.max(np.sqrt(np.sum(ipt ** 2, axis=1)), 0)
    ipt = ipt / var  # scale
    ipt = np.expand_dims(ipt, 0)
    ipt = torch.from_numpy(ipt).float()
    return mean, var

class AdvData_Dataset(Dataset):
    """
    Args:
        csv_path (string): Path to the csv file with annotations.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_path, number_class, original_idx, normalization_flag:int, shuffle_flag: int):
        self.path = data_path
        self.num_of_classes = number_class
        self.len = len(os.listdir(self.path))
        self.ori = original_idx
        self.normal_flag = normalization_flag
        self.shuffle_flag = shuffle_flag
        """
        csv_path = os.path.expanduser(csv_path)
        assert os.path.exists(csv_path), '%s not found' % csv_path

        # read csv file
        self.df = pd.read_csv(csv_path, header=0, names=['point_cloud_path', 'cls_name'])
        self.df.cls_name, _ = pd.factorize(self.df.cls_name, sort=True)

        self._num_of_classes = len(self.df.cls_name.drop_duplicates())
        self._len_of_dataset = len(self.df.cls_name)
        self._transform = transform
        """
    def get_num_of_classes(self):
        return self.num_of_classes

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        """
        point_cloud_path,  cls_id = self.df.iloc[idx]
        _, _, point_cloud_data = read_bntfile(point_cloud_path)
        #pcd = o3d.io.read_point_cloud(point_cloud_path)
        #pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, 4)
        #print("pcd；  ", pcd)
        """
        point_cloud_data, tar = self.read_PC(idx)
        ori = self.ori

        return point_cloud_data, ori, tar
    
    # def read_PC(idx, self.path):
    #     A = zeros((4000, 3), dtype=float)
    #     files = os.listdir(self.path)
    #     ori = tar = idx
    #     for file in files:
    #         id = file.split('-')
    #         if int(id[0]) == idx:
    #             ori = int(id[1])
    #             tar = int(id[2].split('.')[0])
    #             f = open(self.path + "/" + file)
    #             lines = f.readlines()
    #             A_row = 0
    #             for line in lines:
    #                 list = line.strip('\n').split(' ')
    #                 A[A_row:] = list[0:3]
    #                 A_row += 1
    #             break
    #     return A, ori, tar
    def read_PC(self, idx):
        A = zeros((4000, 3), dtype=float)
        files = os.listdir(self.path)

        file = files[idx]
        id = file.split('_')
        tar = int(id[-1].split('.')[0])
        f = open(self.path + "/" + file)
        lines = f.readlines()
        A_row = 0
        for line in lines:
            list = line.strip('\n').split(' ')
            A[A_row:] = list[0:3]
            A_row += 1
        if self.normal_flag==1:
            mean = np.expand_dims(np.mean(A, axis=0), 0)
            A = A - mean
            var = np.max(np.sqrt(np.sum(A ** 2, axis=1)), 0)
            A = A / var 
        elif self.normal_flag==2:
            point_cloud_txt_file_name = 'face0424.txt'
            origin_data_path = os.path.expanduser("~//yq_pointnet//test_face_data/" + point_cloud_txt_file_name)
            mean, var = get_mean_var(origin_data_path)
            A = A - mean
            A = A / var 
        if self.shuffle_flag:
            A = rand_row(A, 4000)
        return A, tar














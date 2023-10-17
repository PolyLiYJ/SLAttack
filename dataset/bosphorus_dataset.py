import os
import sys
import random

sys.path.append('../')
import numpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataset.readbnt import read_bntfile
random.seed(7122)
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
data_root = os.path.expanduser('~/dataset/BosphorusDB')

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

"""
Input:
    xyz: pointcloud data, [B, N, 3]
    npoint: number of samples
Return:
    centroids: sampled pointcloud index, [B, npoint]
"""
def FarthestPointSampling_ForBatch(xyz, npoint):
    B, N, C = xyz.shape
 
    centroids = np.zeros((B, npoint))
    distance = np.ones((B, N)) * 1e10
 
    batch_indices = np.arange(B)
 
    barycenter = np.sum((xyz), 1)
    barycenter = barycenter/xyz.shape[1]
    barycenter = barycenter.reshape(B, 1, 3)
 
    dist = np.sum((xyz - barycenter) ** 2, -1)
    farthest = np.argmax(dist,1)
 
    for i in range(npoint):
        # print("-------------------------------------------------------")
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype('int')


def index_points(points, idx):
    batch_indices = np.arange(points.shape[0])
    batch_indices = batch_indices.repeat(idx.shape[1]).reshape((-1,idx.shape[1]))  #(batch_size,npoints)
   # print(batch_indices.shape)
    new_points = points[batch_indices,idx, :]
    return new_points


class Bosphorus_Dataset(Dataset):
    """
    Args:
        csv_path (string): Path to the csv file with annotations.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_path, transform=None, device="cpu"):
        csv_path = os.path.expanduser(csv_path)
        assert os.path.exists(csv_path), '%s not found' % csv_path

        # read csv file
        self.df = pd.read_csv(csv_path, header=0, names=['point_cloud_path', 'cls_name'])
        self.df.cls_name, _ = pd.factorize(self.df.cls_name, sort=True)

        self._num_of_classes = len(self.df.cls_name.drop_duplicates())
        self._len_of_dataset = len(self.df.cls_name)
        self._transform = transform
        self.device = device

    def get_num_of_classes(self):
        return self._num_of_classes

    def __len__(self):
        return self._len_of_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        point_cloud_path,  cls_id = self.df.iloc[idx]
        point_cloud_data = np.loadtxt(point_cloud_path, delimiter=' ')
        # if int(cls_id) == 105:
        #     point_cloud_data = np.loadtxt(point_cloud_path, delimiter=',')
        #     #point_cloud_data =  farthest_point_sample([point_cloud_data], 4000)
        #     point_cloud_data = point_cloud_data[:, 0:3]
        # elif int(cls_id) == 106:
        #     point_cloud_data = np.loadtxt(point_cloud_path, delimiter=',')
        #     # new_row = [0,0,0,0,0]
        #     # for k in range(2000):
        #     #  point_cloud_data = numpy.vstack([point_cloud_data, new_row])
        #     point_cloud_data = numpy.vstack([point_cloud_data, point_cloud_data[0:4000-np.size(point_cloud_data,axis =0)]])
        #     point_cloud_data = point_cloud_data[:, 0:3]
        # else:
        #     _, _, point_cloud_data = read_bntfile(point_cloud_path)
        #     point_cloud_data =  farthest_point_sample([point_cloud_data], 4000)
        #pcd = o3d.io.read_point_cloud(point_cloud_path)
        #pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, 4)
        #print("pcd；  ", pcd)

        if numpy.any(numpy.isnan(point_cloud_data)):
            point_cloud_data[np.isnan(point_cloud_data)] = 0
        
        point_cloud_data = point_cloud_data - np.expand_dims(np.mean(point_cloud_data, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_cloud_data ** 2, axis = 1)),0)
        point_cloud_data = point_cloud_data / dist #scale
        # point_cloud_data = np.array([point_cloud_data])
        # centriod =  FarthestPointSampling_ForBatch(point_cloud_data, 4000)
        # point_cloud_data =  index_points(point_cloud_data, centriod)
        # point_cloud_data = point_cloud_data[0]
        point_cloud_data = torch.from_numpy(point_cloud_data.astype(np.float)).to(self.device)
        cls_id = torch.from_numpy(np.array([cls_id]).astype(np.int64)).to(self.device)
        return point_cloud_data, cls_id

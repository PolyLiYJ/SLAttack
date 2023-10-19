"""Targeted point perturbation attack."""

import os
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import sys
random.seed(7122)
from CW_utils.basic_util import str2bool, set_seed
from CW_SL_attack import CW
from CW_utils.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss
from CW_utils.dist_utils import L2Dist
from CW_utils.dist_utils import ChamferDist, ChamferkNNDist
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN
from dataset.bosphorus_dataset import Bosphorus_Dataset
import os


def check_num_pc_changed(adv, ori):
    logits_mtx = np.logical_and.reduce(adv == ori, axis=1)
    return np.sum(logits_mtx == False)

def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

# input: numpy.array [K,3]
def preprocess(ipt):
    # normalization
    ipt= ipt[:,0:3]
    ipt = rand_row(ipt, 4000)
    mean = np.expand_dims(np.mean(ipt, axis=0), 0)
    ipt = ipt - mean
    var = np.max(np.sqrt(np.sum(ipt ** 2, axis=1)), 0)
    ipt = ipt / var  # scale
       
    # to tensor
    ipt = np.expand_dims(ipt, 0)
    ipt = torch.from_numpy(ipt).float().to(device).requires_grad_(False) 
    ipt = ipt.permute(0,2,1) #[1,3,4000]
    return ipt, mean, var

def train_one_epoch(epoch_index, training_loader, loss_fn):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        pt = data[0].permute(0,2,1).float().to(device)
        label = data[1].to(device)
        pred,_,_ = model(pt)
        loss = loss_fn(pred,label.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 0:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

if __name__ == "__main__":
     # Training settings
    parser = argparse.ArgumentParser(description='Structured light attack')
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate')    
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS',help='Size of batch')    
    parser.add_argument('--binary_step', type=int, default=5, metavar='N',help='Binary search step')
    parser.add_argument('--data_root', type=str,default='data/attack_data.npz')    
    parser.add_argument('--dataset', type=str, default='Bosphorus', help="dataset: Bosphorus | Eurecom")    
    parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')     
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--feature_transform', type=str2bool, default=False,help='whether to use STN on features in PointNet')    
    parser.add_argument('--gpu', type=int, default=0,help='gpu index')   
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training') 
    parser.add_argument('--model', type=str, default='PointNet++Msg', metavar='N',choices=['PointNet', 'PointNet++Msg', 'PointNet++Ssg',
                                 'DGCNN'],help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--num_points', type=int, default = 1024,help='num of points to use')
    parser.add_argument('--continue_train',action='store_true')


    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    '''
    set parameters
    '''
    count = 0
    num = 0
    label = []
    
    '''
    load model
    '''
    if args.dataset == 'Bosphorus':
        num_of_class = 107
    elif args.dataset == 'Eurecom':
        num_of_class = 52

    if args.model == 'PointNet':
        model = PointNetCls(k=num_of_class, feature_transform=False)    
    elif args.model == 'PointNet++Msg':
        model = PointNet_Msg(num_of_class, normal_channel=False)
    elif args.model == 'PointNet++Ssg':
        model = PointNet_Ssg(num_of_class)
    elif args.model == 'DGCNN':
        model = DGCNN(args, output_channels=num_of_class)
    else:
        exit('wrong model type')        
    
    if args.continue_train:
        state_dict = torch.load(f'cls/{args.dataset}/{args.model}_model_on_{args.dataset}_best.pth')
        model.load_state_dict(state_dict)
    model.to(device)
    train_dataset = Bosphorus_Dataset(csv_path="dataset/train_farthest_sampled.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    
    test_dataset = Bosphorus_Dataset(csv_path="dataset/test_farthest_sampled.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    print("length of train dataset:", len(train_dataset))
    print("length of test dataset:", len(test_dataset))
    
    model.eval()
    total_num = 0
    total_success_num = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    epoch_number = 0
    EPOCHS = 50
    best_vloss = 1_000_000.
    best_acc = 0
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, train_dataloader, loss_fn)

        model.eval()
        running_vloss = 0.0
        total_num = 0
        correct_num = 0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_dataloader):
                # Every data instance is an input + label pair
                vpt = vdata[0].permute(0,2,1).float().to(device)
                vlabel = vdata[1].to(device)
                vpred,_,_ = model(vpt)
                vloss = loss_fn(vpred, vlabel.squeeze(1))
                running_vloss += vloss
                total_num = total_num + args.batch_size
                correct_num = correct_num + np.sum(torch.argmax(vpred, dim = 1).cpu().numpy()==vlabel.cpu().numpy())
                

        avg_vloss = running_vloss / (i + 1)
        acc = correct_num*1.0 / total_num
        print('LOSS train {} valid {} test acc{}'.format(avg_loss, avg_vloss, acc))


        # Track best performance, and save the model's state
        if acc > best_acc:
            best_acc = acc
            model_path = f'cls/{args.dataset}/{args.model}_model_on_{args.dataset}_best.pth'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
            
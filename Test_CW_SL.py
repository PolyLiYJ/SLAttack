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
device = torch.device("cpu")
def check_num_pc_changed(adv, ori):
    logits_mtx = np.logical_and.reduce(adv == ori, axis=1)
    return np.sum(logits_mtx == False)

def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

def CW_attack_api(args, pt, label = 105):

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
        model = DGCNN(args, output_channels=num_of_class).to(device)
    else:
        exit('wrong model type')        
        
    model.load_state_dict(
        torch.load(f'cls/{args.dataset}/{args.model}_model_on_{args.dataset}.pth', map_location=torch.device('cpu')) )

    model.to(device)

    # setup attack settings

    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa, whether_target=args.whether_target, device = device)
    else:
        adv_func = CrossEntropyAdvLoss().to(device)
    

    # run attack
    model.eval()
    # hyper-parameters from their official tensorflow code
    attacker = CW(model, adv_func, args.dist_function,
                  attack_lr=args.attack_lr,
                  init_weight=0.1, max_weight=100.0,
                  binary_step=args.binary_step,
                  num_iter=args.num_iter,
                  whether_target=args.whether_target,
                  whether_1d=args.whether_1d)

    # attack
    if args.dist_function in {'Chamfer_pt', 'L2Loss_pt', 'ChamferkNN_pt'}:
        adv_PC, susscessnum, x_p, pred = attacker.attack_optimize_on_pointcloud(pt, label, args)
    else:
        adv_PC, susscessnum, x_p, pred = attacker.attack_optimize_on_yp(pt, label, args)
    
    return adv_PC, susscessnum, x_p, pred

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



if __name__ == "__main__":
     # Training settings
    parser = argparse.ArgumentParser(description='Structured light attack')
    parser.add_argument('--adv_func', type=str, default='logits',choices=['logits', 'cross_entropy'],help='Adversarial loss function to use')
    parser.add_argument('--attack_lr', type=float, default=0.01,help='lr in CW optimization')    
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',help='Size of batch')    
    parser.add_argument('--binary_step', type=int, default=5, metavar='N',help='Binary search step')
    parser.add_argument('--data_root', type=str,default='data/attack_data.npz')    
    parser.add_argument('--dataset', type=str, default='Bosphorus', help="dataset: Bosphorus | Eurecom")    
    parser.add_argument('--dist_function', default="L2Loss", type=str,
                        help=' L1Loss, L2Loss, L2Loss_pt, Chamfer_pt, ChamferkNN_pt')
    parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')    
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--feature_transform', type=str2bool, default=False,help='whether to use STN on features in PointNet')    
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--kappa', type=float, default=10.,help='min margin in logits adv loss')   
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training') 
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',choices=['PointNet', 'PointNet2_MSG', 'PointNet2_SSG',
                                 'DGCNN'],help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--num_points', type=int, default = 4000,help='num of points to use')
    parser.add_argument('--num_iter', type=int, default= 100, metavar='N',help='Number of iterations in each search step')
    parser.add_argument('--normal_name', default='person1.txt', type=str,help='The normal point cloud filename')
    
    parser.add_argument('--whether_1d', default=True, type=bool,help='True for z perturbation, False for xyz perturbation')
    parser.add_argument('--whether_target', default=False, action='store_true', help='True for target attack, False for untarget attack')
    parser.add_argument('--whether_renormalization', default=True, type=bool,
                        help='True for renormalization')
    parser.add_argument('--whether_3Dtransform', default=True, type=bool,
                        help='3D transformation')
    parser.add_argument('--whether_resample', default=True, type=bool,
                        help='whether resample using farest sampling')
    args = parser.parse_args()
    '''
    set parameters
    '''
    data_root = os.path.expanduser("test_face_data/")
    normalPC = os.path.join(data_root, args.normal_name)
    count = 0
    num = 0
    label = []
    
    '''
    load model
    '''
    model = PointNetCls(k=107, feature_transform=False)
    model.load_state_dict(
        torch.load(f'cls/{args.dataset}/{args.model}_model_on_{args.dataset}.pth', map_location=torch.device('cpu')) )
    model.to(device)
    test_data_path = os.path.expanduser(data_root + args.normal_name)
    try:
        pt_ori = np.loadtxt(test_data_path, delimiter=' ')
    except:
        pt_ori = np.loadtxt(test_data_path, delimiter=',')
    pt_normalized,mean, var = preprocess(pt_ori)
    model.eval()
    pred,_,_ = model(pt_normalized)
    originalLabel = torch.argmax(pred).cpu().numpy()
    print("the original label of test data:", originalLabel)
    print("Target attack:", args.whether_target)

    if args.whether_target == False:
        advPC, successnum, x_p_rebuild, out_prediction = CW_attack_api(args, pt = pt_ori, label= originalLabel)
        adv_fname = os.path.join(data_root, 'adv_' + args.normal_name[:-4] + "_untargeted_" + args.dist_function + "_"+ str(originalLabel) +'.txt')
        np.savetxt(adv_fname,advPC, fmt='%.04f', delimiter = ',')
        print("save file at", adv_fname)
        adv_fname = os.path.join(data_root, 'adv_' + args.normal_name[:-4] + "_untargeted_" + args.dist_function + '_' + str(originalLabel) + '_xp.txt')
        np.savetxt(adv_fname,x_p_rebuild)
    else:
        targetLabel = random.randint(0, 100)
        print("Target label:", targetLabel)
        if targetLabel != originalLabel:
            advPC, successnum, x_p_rebuild, out_prediction = CW_attack_api(args, pt = pt_ori, label = targetLabel)
            adv_fname = os.path.join(data_root, 'adv_' + args.normal_name[:-4] + "_targeted_" + str(targetLabel) + '.txt')
            np.savetxt(adv_fname, advPC, fmt='%.04f', delimiter = ',')
            print("save adversarial point cloud at", adv_fname)
            adv_fname = os.path.join(data_root, 'adv_' + args.normal_name[:-4] + "_targeted_" + str(targetLabel) + 'xp.txt')
            np.savetxt(adv_fname,x_p_rebuild)
            print("save adversarial phase map at", adv_fname)
            pt_normalized, _,_ = preprocess(advPC)
            pred, _, _ = model(pt_normalized)
            predLabel = torch.argmax(pred).cpu().numpy()
            print("the pred label of adv data:", predLabel)



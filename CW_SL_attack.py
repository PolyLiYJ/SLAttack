
import random
import torch
import torch.optim as optim
import numpy as np
import scipy.io as scio
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.linalg import det
from CW_utils.dist_utils import L2Dist, Smooth_loss,L2_Attention,Lhalf_Attention
from CW_utils.dist_utils import ChamferDist, ChamferkNNDist
from math import exp,pow,sqrt
from Visualization import showXYZ, mesh, plot,  gaussBlur
import cv2
device = torch.device("cpu")
out_root = 'test_face_data'

class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0,max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

clamp_class = Clamp()
clamp = clamp_class.apply

class Clamp1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, x_0):
        N = 12 
        delta = 1/float(N)
        return torch.max(torch.min(input.clamp(min=0,max=1), x_0.detach()+delta), x_0.detach()-delta)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone() 
        return grad_input, None
clamp_class1 = Clamp1()
clamp1 = clamp_class1.apply

# customed gradient backward function, see https://pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html
class reconstruct3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_p, face_area, Ac, Ap):        
        [y, x, w, h]= face_area
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        xyz = []

        for vc in range(x,x+h):
            for uc in range(y, y+w):        
                if y_p[vc,uc]==0:
                    pass
                else:
                    A = torch.cat((Ac[0:2,0:3] - torch.cat((Ac[2,0:3].unsqueeze(0) *uc,Ac[2,0:3].unsqueeze(0) *vc),dim=0),
                                Ap[0,0:3].unsqueeze(0) - Ap[2,0:3].unsqueeze(0)  * y_p[vc,uc]),dim=0)  
                    b = torch.stack((Ac[2,3] * uc  - Ac[0,3], 
                                    Ac[2,3] * vc- Ac[1,3],
                                    Ap[2,3] * y_p[vc,uc] - Ap[0,3]),dim=0)
                    A = A.cpu().numpy()
                    b = b.cpu().numpy()
                    point = np.matmul(np.linalg.inv(A),b.T)
                    xyz.append(point)
    
        xyz_rebuild = torch.tensor(np.array(xyz)).float().to(device) #[4200,3], [x,y,z,uc,vc]
        ctx.save_for_backward(y_p, xyz_rebuild)
        ctx.Ac = Ac
        ctx.Ap = Ap
        ctx.face_area = face_area
        return xyz_rebuild
       
       
    #input ： gradient of dL_dz, [K,3]    
    @staticmethod
    def backward(ctx, grad_output):
       # np.savetxt("output/grad_output.txt", grad_output.cpu().numpy())
        y_p, xyz_rebuild = ctx.saved_tensors
        grad_yp = torch.tensor(np.zeros(y_p.shape), dtype=torch.float).to(device)
        for i in range(0, xyz_rebuild.size(dim=0)):
            point =  xyz_rebuild[i,:].unsqueeze(1) #[3,1]
            x = point[0,0]
            y = point[1,0]
            z = point[2,0]
            extend_point = torch.cat((point, torch.tensor([[1]], dtype=torch.float).to(device)), dim =0)
            usv_c = torch.matmul(ctx.Ac,extend_point ) # the 
            # vertical coordinate
            x_c = torch.round(usv_c[1]/usv_c[2]).long()
            # horizontal coordinate
            y_c = torch.round(usv_c[0]/usv_c[2]).long()            
            # x_cor = min(max(int(x_c.cpu().numpy()[0]),0),height-1)
            # y_cor = min(max(int(y_c.cpu().numpy()[0]),0), width-1)
            if x_c < ctx.face_area[1]+ctx.face_area[3] and x_c >= ctx.face_area[1] and y_c < ctx.face_area[0]+ctx.face_area[2] and y_c >=ctx.face_area[0]:
                x_cor = x_c
                y_cor = y_c
                a = ctx.Ap[0,2]
                b = ctx.Ap[0,0]*x + ctx.Ap[0,1]*y+ctx.Ap[0,3]
                c = ctx.Ap[2,2]
                d = ctx.Ap[2,0]*x + ctx.Ap[2,1]*y+ctx.Ap[2,3]
                dz_du =(a*d-b*c)/torch.pow((c*y_p[x_cor, y_cor]-a),2)
                # approxiate through the z direction gradient
                grad_yp[x_cor,y_cor] = grad_output[i,2] * dz_du
        # mesh([100 , 38,  57 , 90],grad_yp.cpu().numpy(),"output/grad_yp.png")
        # np.savetxt("output/grad_yp.txt", grad_yp.cpu().numpy()[38:39+90,100:100+57])
        return grad_yp.clone(), None, None, None

# another way of 3D reconstruct
def rebuild3D(y_p_rebuild, face_area, Ac, Ap):
    # here we only caculate the gradient on vector b, because caculating the grad of A and b will confict on the y_p_rebuild's grad
    A = [[torch.cat(((torch.FloatTensor(Ac[0:2,0:3]).to(device) - torch.cat((torch.FloatTensor(Ac[2,0:3]).unsqueeze(0).to(device) *uc,
                                                            torch.FloatTensor(Ac[2,0:3]).unsqueeze(0).to(device) *vc),dim=0)),
                    torch.FloatTensor(Ap[0,0:3]).unsqueeze(0).to(device) - torch.FloatTensor(Ap[2,0:3]).unsqueeze(0).to(device)  * y_p_rebuild[vc,uc].detach()),dim=0)
        .float().to(device).requires_grad_(False) for uc in range(face_area[0],face_area[0]+face_area[2])]
        for vc in range(face_area[1],face_area[1]+face_area[3]) ]
    
    b = [[torch.stack(((torch.FloatTensor([Ac[2,3] * uc]).to(device)  - torch.FloatTensor([Ac[0,3]]).to(device)), 
                    (torch.FloatTensor([Ac[2,3] * vc] ).to(device)- torch.FloatTensor([Ac[1,3]]).to(device)),
                    (torch.FloatTensor([Ap[2,3]]).to(device) * y_p_rebuild[vc,uc] - torch.FloatTensor([Ap[0,3]]).to(device))),dim=0)
        .float().to(device).detach().requires_grad_(False) for uc in range(face_area[0],face_area[0]+face_area[2]) ] 
        for vc in range(face_area[1],face_area[1]+face_area[3])]
    
    
    xyz_rebuild =  [torch.cat((torch.mm(torch.linalg.inv(A[vc][uc]),b[vc][uc]).t(),
                            torch.tensor([vc, uc]).unsqueeze(0).float().to(device)), dim = 1).requires_grad_(False) 
                    for uc in range(face_area[2]) for vc in range(face_area[3])]

    xyz_rebuild2 = torch.stack(xyz_rebuild,dim = 0).squeeze(1) #[4200,5], [x,y,z,uc,vc]
    [x,y,*notused] = torch.chunk(xyz_rebuild2, 5, dim = 1)

    up =  y_p_rebuild[face_area[1]:face_area[1]+face_area[3],face_area[0]:face_area[0]+face_area[2]].permute(1,0).reshape((-1,)).unsqueeze(1)
    z = torch.div(Ap[0,0]*x.detach()+Ap[0,1]*y.detach()+Ap[0,3]-torch.mul((Ap[2,0]*x.detach()+Ap[2,1]*y.detach()+Ap[2,3]),up),Ap[2,2]*up-Ap[0,2])
    xyz_rebuild3 = torch.cat((x.detach(), y.detach(),z), 1)
    showXYZ(xyz_rebuild3[:,0:3].cpu().detach().numpy(), "output/xyz_rebuild.png")
    return xyz_rebuild3


def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))
 
def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)
 
def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1])).float().to(device)
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result
 
def invmat(M):
    return 1.0/det(M)*adj(M)

class CW:
    """Class for CW attack.
    """

    def __init__(self, model, adv_func, dist_func, attack_lr=1e-2,
                 init_weight=1e-3, max_weight=100., binary_step=10, num_iter=500, whether_target=True, whether_1d=True, abort_early=False):
        """CW attack by perturbing points.
        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            init_weight (float, optional): weight factor init. Defaults to 10.
            max_weight (float, optional): max weight factor. Defaults to 80.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
        """

        self.model = model.to(device)
        self.model.eval()

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.whether_target = whether_target
        self.whether_1d = whether_1d
        self.abort_early = abort_early
        
        self.n = 4 # digital number of Gray code
        self.N = 12. # order of gray code
        # load camera extric matrix
        self.height = int(180)*4
        self.width = int(320)*4     
        RTc = scio.loadmat('test_face_data/CamCalibResult.mat');
        self.Kc = RTc['KK']
        self.Rc_1 = RTc['Rc_1']
        self.Tc_1 = RTc['Tc_1']
        #Kc = np.dot([[0.25,0,0],[0,0.25,0],[0,0,1]],Kc) #resize the image to 0.25*width
        self.Ac = np.dot(self.Kc, np.append(self.Rc_1, self.Tc_1,axis = 1))
        # load projector extric matrix
        RTp = scio.loadmat('test_face_data/PrjCalibResult.mat');
        self.Kp = RTp['KK']
        self.Rp_1 = RTp['Rc_1']
        self.Tp_1 = RTp['Tc_1']
        #Kp = np.dot([[0.25,0,0],[0,0.25,0],[0,0,1]],Kp)
        self.Ap = np.dot(self.Kp, np.append(self.Rp_1, self.Tp_1,axis = 1))

    
    def attack_optimize_on_yp(self, unnormalized_data,label, args):
        
        """Attack on given data to target.
        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        
        Xws = unnormalized_data[:,0]
        Yws = unnormalized_data[:,1]
        Zws = unnormalized_data[:,2]
        
        area = np.zeros((self.height, self.width),dtype=float)
        y_p = np.full((self.height, self.width),0.0)
        
        count_xy = 0 
        xcol = []
        ycol = []
        
        #read data and compute corresponding camera and projector cooridates
        for i in range(0,unnormalized_data.shape[0]):
            if np.isnan(Xws[i]) == False:
                count_xy = count_xy+1;
                point = np.array([Xws[i],Yws[i],Zws[i],1])
                usv = np.dot(self.Ap, point.T).astype(float) # the 
                usv_c = np.dot(self.Ac, point.T).astype(float)
                # vertical coordinate
                x = round(usv_c[1]/usv_c[2])
                # horizontal coordinate
                y = round(usv_c[0]/usv_c[2])
                xcol.append(x)
                ycol.append(y)
                y_p[x,y] = usv[0]/usv[2] # the horizontal coordinate in projector for (x,y) pixel in the captured image
                area[x,y] = 1.0
        
        xcol = np.array(xcol)
        ycol = np.array(ycol)
        face_area = np.array([np.min(ycol),np.min(xcol), np.max(ycol)-np.min(ycol), np.max(xcol)-np.min(xcol)],dtype=np.uint32)

        atten_map = self.get_attention_map(face_area,  mask = area, out_root=out_root, y_p = y_p)
        atten_map = torch.from_numpy(atten_map).to(device)
        atten_map.requires_grad = False
        B = 1 # batchsize
                         
        # weight factor for budget regularization
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight
        
        
        # perform binary search
        
        o_bestdist = np.array([1e10] * B)
        Ac = torch.tensor(self.Ac, dtype = torch.float, requires_grad=False).to(device)
        Ap = torch.tensor(self.Ap, dtype = torch.float, requires_grad=False).to(device)   
        mask = torch.tensor(area,dtype = torch.float, requires_grad=False).to(device) 

        for binary_step in range(self.binary_step):
            
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            adv_loss = torch.tensor(0.).to(device)
            dist_loss = torch.tensor(0.).to(device)
            pre_loss = 1e10 # used for early break
            
            phaX_rebuild = torch.tensor(y_p / self.width, dtype=torch.float).to(device)
            phaX_rebuild.requires_grad = False#[0,1]
            Noise = torch.zeros(np.shape(y_p), dtype=torch.float).to(device)
            Noise.requires_grad = True
            
            opt = optim.Adam([Noise], lr=self.attack_lr, weight_decay=0.)  

            ori_phaX_rebuild = torch.tensor(y_p / self.width, dtype=torch.float).to(device)
            ori_phaX_rebuild.requires_grad = False
            
            attacked_label = torch.tensor(np.array([label]), dtype=torch.float).to(device)
            attacked_label.requires_grad = False
            # one step in binary search
            for iteration in range(self.num_iter):
                phaX_rebuild_add_noise = phaX_rebuild + Noise
                # clip pha into [max(0,pha-delta), max(1,pha+delta)]
                clamp1 = Clamp1.apply
                phaX_rebuild_clamped = clamp1(phaX_rebuild_add_noise, ori_phaX_rebuild)
                
                phaX_rebuild_masked = phaX_rebuild_clamped * mask
                y_p_rebuild = phaX_rebuild_masked * self.width

                Rebuild3D = reconstruct3D.apply
                xyz_rebuild = Rebuild3D(y_p_rebuild,face_area, Ac, Ap)

                adv_data = xyz_rebuild[:,0:3].permute(1,0).to(device) #[3,K]                

                # align the point cloud with the camera by rotate the point cloud, see equation
                KcRc = torch.tensor(np.dot(self.Kc,self.Rc_1), requires_grad=False).float().to(device)
                adv_data_aligned = torch.matmul(KcRc, adv_data).unsqueeze(0)   #[B,3,K]
                
                # renormalization
                # adv_data_normed, mean, var = self.normalize(adv_data_aligned)
   
                if iteration == 0 :
                    ori_data_backed = adv_data.clone().detach().unsqueeze(0) #[B,3,K]
                    ori_data_backed.requires_grad = False
                    ori_data_normed,_,_ = self.normalize(ori_data_backed) #[B,3,K]
                    ori_data_normed.requires_grad = False
                    ori_data_aligned =  adv_data_aligned.clone().detach() # [B,3,K]
                    ori_data_aligned.requires_grad = False
                
                if args.whether_1d:
                    # single direction constraint   , rotate the point cloud
                    adv_z = adv_data_aligned[0,2,:].unsqueeze(0).unsqueeze(1) #[B,1,K]
                    # add noise to the rotated point cloud
                    adv_data_cat = torch.cat((ori_data_aligned[0,0:2,:].unsqueeze(0), adv_z), dim = 1)
                else:
                    adv_data_cat = adv_data_aligned
                    
                # recover the point cloud
                inv_KcRc = torch.tensor(np.matmul(np.linalg.inv(self.Rc_1), np.linalg.inv(self.Kc)), requires_grad=False).float().to(device)
                adv_data_backed = torch.matmul(inv_KcRc, adv_data_cat[0]).unsqueeze(0)
                
                # compute distance loss       
                if self.dist_func == 'SmoothLoss':
                    dist_loss = Smooth_loss(Noise, mask)
                elif self.dist_func == 'L1Loss':
                    dist_loss = torch.nansum(torch.abs(Noise)+1e-10,(0,1))
                elif self.dist_func == 'L2Loss':
                    dist_loss = torch.nansum((Noise) ** 2 * mask, dim=[0,1])
                elif self.dist_func == 'L2Loss_attention':
                    dist_loss = torch.nansum((Noise) ** 2 * mask * atten_map, dim=[0,1])
                elif self.dist_func == 'L2_L1':
                    L2Loss = torch.nansum((Noise) ** 2, dim=[0,1])
                    L1Loss = torch.nansum(torch.abs(Noise)+1e-10,(0,1))
                    dist_loss =   L2Loss +  L1Loss
                    
                    
                if args.whether_renormalization == True:     
                    adv_data_normed, _ , _ = self.normalize(adv_data_backed)
                else:
                    adv_data_normed = adv_data_backed

                if args.whether_3Dtransform == True:
                    adv_loss_list = torch.empty(10)
                    for i in range(10):
                        adv_data_rotated = self.randomRotation(adv_data_normed, ori_data_normed)
                        logits,_,_ = self.model(adv_data_rotated)
                        adv_loss_list[i] = self.adv_func(logits, attacked_label ).mean()
                    adv_loss = torch.mean(adv_loss_list)
                    loss = adv_loss + torch.mul( dist_loss , current_weight[0])
                else:
                    logits,_,_ = self.model(adv_data_normed)
                    adv_loss = self.adv_func(logits,attacked_label).mean()
                    loss = adv_loss + torch.mul(dist_loss , current_weight[0]) 
                    
                pred = torch.argmax(logits, dim=1)  
                softmax = torch.nn.Softmax(dim=1)
                score = softmax(logits)[0,int(label)]  
                
                loss.backward()
                with torch.no_grad():
                    opt.step()    
                    opt.zero_grad()
                
                dist_val = [dist_loss.detach().cpu().numpy()]
                adv_val = [adv_loss.detach().cpu().numpy()]
                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = [xyz_rebuild.detach().cpu().numpy()]  # [B,K,3]
                score_val = score.detach().cpu().numpy()
                # update
                for e, (dist, pred, ii) in \
                        enumerate(zip(dist_val, pred_val,input_val)):
                    if args.whether_target == True:
                        flag = (pred == attacked_label)
                    else:
                        flag = (pred != attacked_label)
                    if dist < bestdist[e] and flag:
                        bestdist[e] = dist
                        bestscore[e] = score_val
                    if dist < o_bestdist[e] and flag:
                        o_bestdist[e] = dist
                        o_bestattack = ii
                        o_best_pred = pred
                        o_best_yp_rebuild = y_p_rebuild.detach().cpu().numpy()

                # mesh(face_area, pha_wrapped.cpu().detach().numpy())
                if iteration % 100 == 0:
                    print("binary step:", binary_step, "  iteration:", iteration, "current weight:", current_weight[0])
                    print("loss: %2.5f, dist loss: %2.5f, adv_loss: %2.5f , pred: %d" % ( loss.item(),dist_loss.item(), adv_loss.item(),  pred_val))
                    
                if self.abort_early and not iteration % (self.num_iter // 10):
                    if loss.item() > pre_loss * 0.999:
                        break
                    pre_loss = loss.detach().cpu().numpy()

                    
            for e, label in enumerate(attacked_label.cpu().numpy()):
                if bestscore[e] != -1 and bestdist[e] <= o_bestdist[e] :
                    # success, increase the weight,
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.

        # end of CW attack
        # fail to attack some examples
        # just assign them with last time attack data

        # return final results
        success_num = (lower_bound > 0.).sum()
        if success_num == 0:
            fail_idx = (lower_bound == 0.)
            #o_bestattack = input_val[fail_idx]
            o_bestattack = None
            o_best_yp_rebuild = None, 
            o_best_pred = None
            print("Fail to attack")
        else:
            print('Successfully attack {}/{}   pred: {}  best dist loss : {}, best weight: {} '.format(success_num, B, o_best_pred, o_bestdist[0], current_weight[0] ))
        return o_bestattack, success_num, o_best_yp_rebuild, o_best_pred
    
    """Attack the point cloud on given data to target.
    Args:
        unnormalized_data (torch.FloatTensor): victim data, [B, num_points, 3]
        label (torch.LongTensor): target label for target attack or original label for untorget attack, [B]
        outfolder: The output folder
        args: argments
    """  
    def attack_optimize_on_pointcloud(self, unnormalized_data,label, args):
        # load camera extric matrix
        RTc = scio.loadmat('test_face_data/CamCalibResult.mat')
        Kc = RTc['KK']   # camera intrinsics
        Rc_1 = RTc['Rc_1']
        Tc_1 = RTc['Tc_1']
        Kc = np.dot([[0.5,0,0],[0,0.5,0],[0,0,1]],Kc) #resize the image
        Ac = np.dot(Kc, np.append(Rc_1, Tc_1,axis = 1))
        # load projector extric matrix
        RTp = scio.loadmat('test_face_data/PrjCalibResult.mat')
        Kp = RTp['KK']   # projector intrinsics
        Rp_1 = RTp['Rc_1']
        Tp_1 = RTp['Tc_1']
        Kp = np.dot([[0.5,0,0],[0,0.5,0],[0,0,1]],Kp)
        Ap = np.dot(Kp, np.append(Rp_1, Tp_1,axis = 1))
        
        
        Xws = unnormalized_data[:,0]
        Yws = unnormalized_data[:,1]
        Zws = unnormalized_data[:,2]
                                                                      
        n = 4 # digital number of Gray code

        area = np.zeros((self.height, self.width))
        y_p = np.full((self.height, self.width), 0)
        count_xy = 0 
        xcol = []
        ycol = []
        if self.dist_func == 'ChamferkNN_pt':
            dist_func = ChamferkNNDist()
        elif self.dist_func == 'Chamfer_pt':
            dist_func = ChamferDist()  
        elif self.dist_func == 'L2Loss_pt':
            dist_func = L2Dist() 
        
        #read data and compute corresponding camera and projector cooridates
        for i in range(0,unnormalized_data.shape[0]):
            if np.isnan(Xws[i]) == False:
                count_xy = count_xy+1;
                point = np.array([Xws[i],Yws[i],Zws[i],1])
                usv = np.dot(Ap, point.T) # the 
                usv_c = np.dot(Ac, point.T)
                # vertical coordinate
                x = round(usv_c[1]/usv_c[2])
                # horizontal coordinate
                y = round(usv_c[0]/usv_c[2])
                xcol.append(x)
                ycol.append(y)
                y_p[x,y] = usv[0]/usv[2] # the horizontal coordinate in projector for (x,y) pixel in the captured image
                area[x,y] = 1
        
        xcol = np.array(xcol)
        ycol = np.array(ycol)
        face_area = np.array([np.min(ycol),np.min(xcol), np.max(ycol)-np.min(ycol), np.max(xcol)-np.min(xcol)],dtype=np.uint32)

        B = 1 # batchsize
                         
        # weight factor for budget regularization
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight
        
        
        # perform binary search
        
        o_bestdist = np.array([1e10] * B)
        
        phaX_rebuild = torch.from_numpy(y_p / self.width).to(device,dtype=torch.float)
        y_p_rebuild = phaX_rebuild * self.width

        # mesh(face_area, y_p_rebuild.cpu().detach().numpy(), "y_p_rebuild.png")
        Ac = torch.tensor(Ac, dtype = torch.float, requires_grad=False).to(device)
        Ap = torch.tensor(Ap, dtype = torch.float, requires_grad=False).to(device)    
        xyz_rebuild = reconstruct3D.apply(y_p_rebuild,face_area, Ac, Ap) #[K,3]
        KcRc = np.dot(Kc,Rc_1)

        for binary_step in range(self.binary_step):
            
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            bestpred= np.array([label] * B)
            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()
            pre_loss = 1e10
              
            adv_data = torch.tensor(xyz_rebuild.permute(1,0).cpu().numpy()).unsqueeze(0).float().to(device)
            adv_data_normed, center, var= self.normalize(adv_data) 
            ori_data_normed = adv_data_normed.clone().detach() # [B,3,K]
            ori_data_normed.requires_grad = False
            opt = optim.Adam([adv_data_normed], lr=self.attack_lr, weight_decay=0.) 

            # one step in binary search
            for iteration in range(self.num_iter):
                
                # single direction constraint   
                # dist_loss = dist_func(adv_data_cat.permute((0,2,1)), ori_data_normed.permute((0,2,1)))
                dist_loss = dist_func(adv_data_normed.permute((0,2,1)), ori_data_normed.permute((0,2,1)))
                attacked_label = torch.tensor(np.array([label]), dtype=torch.float).to(device).requires_grad_(False)
                if args.whether_renormalization == True: 
                    adv_data_normed,_,_ = self.normalize(adv_data_normed)
                if args.whether_3Dtransform == True:
                    adv_loss_list = torch.empty(10)
                    for i in range(10):
                        adv_data_rotated = self.randomRotation(adv_data_normed, ori_data_normed)
                        #renormalization
                        adv_data_rotated_renormed,_,_ = self.normalize(adv_data_rotated)
                        logits,_,_ = self.model(adv_data_rotated_renormed)
                        adv_loss_list[i] = self.adv_func(logits, attacked_label ).mean()
                    adv_loss = torch.mean(adv_loss_list)
                    loss = adv_loss + torch.mul(dist_loss , current_weight[0])
                else:
                    logits,_,_ = self.model(adv_data_normed)
                    adv_loss = self.adv_func(logits,attacked_label).mean()
                    loss = adv_loss + torch.mul(dist_loss , current_weight[0])

                pred = torch.argmax(logits, dim=1)
                softmax = torch.nn.Softmax(dim=1)
                score = softmax(logits)[0,int(label)]
                    
                loss.backward()
                opt.step()    
                opt.zero_grad()

                dist_val = [dist_loss.detach().cpu().numpy()]
                adv_val = [adv_loss.detach().cpu().numpy()]
                pred_val = pred.detach().cpu().numpy()  # [B]
                # input_val = adv_data_cat.detach().cpu().numpy()  # [B, 3, K]
                input_val = adv_data_normed.detach().cpu().numpy()
                score_val = score.detach().cpu().numpy()
                # update
                for e, (dist, pred, ii) in \
                        enumerate(zip(dist_val, pred_val,input_val)):
                    if args.whether_target == True:
                        flag = (pred == attacked_label)
                    else:
                        flag = (pred != attacked_label)
                    if dist < bestdist[e] and flag:
                        bestdist[e] = dist
                        bestpred[e] = pred
                        bestscore[e] = score_val
                    # if dist < o_bestdist[e] and pred != ori_label:
                    if dist < o_bestdist[e] and flag:
                        o_bestdist[e] = dist
                        o_bestattack = ii
                        pc = o_bestattack.transpose((1,0))
                             
                if iteration % 10 == 0:
                    print("Step:", binary_step, " Iteration:", iteration, " Weight:", current_weight[0])
                    print("loss: %2.2f, dist loss: %2.2f, adv_loss: %2.2f , pred: %d" % (
                    loss.item(), dist_loss.item(), adv_loss.item(), pred_val))
                    
                    
                if self.abort_early and not iteration % (self.num_iter // 10):
                    if loss.detach().cpu().numpy() > pre_loss * 0.99:
                        break
                    pre_loss = loss.detach().cpu().numpy()
            
                   
            # adjust weight factor   
            for e, label in enumerate(attacked_label.cpu().numpy()):
                if bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    # success 
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    # failure
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                    

            torch.cuda.empty_cache()

        # end of CW attack
        # fail to attack some examples
        # just assign them with last time attack data

        # return final results
        success_num = (lower_bound > 0.).sum()
        if success_num == 0:
            fail_idx = (lower_bound == 0.)
            o_bestattack = input_val[fail_idx][0]
            print("Fail to attack ")
        else:
            print('Successfully attack {}/{}   pred: {}, best dist loss: {}'.format(success_num, B, pred, o_bestdist[0] ))
        pc = o_bestattack.transpose((1,0))
        return pc, success_num, y_p_rebuild.cpu().detach().numpy(), bestpred[0]
    
    
    '''
    prediction
    '''
    def pred(self, data):
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        # ori_data.requires_grad = False

        adv_data = ori_data.clone().detach()
        logits, _,_ = self.model(adv_data)  # [B, num_classes]
        pred = torch.argmax(logits, dim=1)
        return pred.item()
    
    # input [B, 3, K]
    def randomRotation(self, data, ori_data):
        diff =  data - ori_data
        theta = torch.randn(1)*1e-2
        Tz = torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                            [-torch.sin(theta), torch.cos(theta), 0],
                            [0,0,1]],dtype = torch.float).unsqueeze(0)
        Ty = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                        [0,1,0],
                            [-torch.sin(theta),0, torch.cos(theta)]],dtype = torch.float).unsqueeze(0)
        Tx = torch.tensor([[1,0,0],
                        [0,torch.cos(theta), torch.sin(theta)],
                            [0, -torch.sin(theta),torch.cos(theta)]], dtype = torch.float).unsqueeze(0)
        To = torch.tensor([[1,0,0],
                        [0,1,0],
                            [0, 0, 1]], dtype =torch.float).unsqueeze(0)
        r = random.random()
        if r<0.2:
            Tr = Tz
        elif r<0.4:
            Tr = Tx
        elif r<0.6:
            Tr = Ty
        else:
            Tr = To
        Tr = Tr.to(device)
        # normal distribution noises
        # zero = torch.randn(((B, 2, K))) * 1e-4 # x and y axis
        # rand = torch.randn(((B, 1, K))) * 1e-2 # z axis
        
        # adv_data2 =  torch.bmm(Tz.detach(), ori_data.detach()) + diff + torch.cat((zero,rand),1).cuda()
        rotated_data =  (torch.bmm(Tr.detach(), ori_data.detach()) + diff)
        return rotated_data
 
    ''' input: advdata,
        size: [B,3,K]
        normalized data in cube with each eadge in [-1,1]
    ''' 
    def normalize(self, adv_data):
        if adv_data.shape[2] != 3:
            adv_data2 = adv_data.permute(0,2,1) #[B,K,3]
        centroid = torch.mean(adv_data2,axis=1).detach()
        adv_data3 = adv_data2 - centroid.unsqueeze(1)
        var = torch.max(torch.sqrt(torch.sum(adv_data3**2,axis=2)),axis=1,keepdim=True)[0].detach()
        adv_data4 = adv_data3/var.unsqueeze(1)
        adv_data5 = adv_data4.permute(0,2,1)
        return adv_data5, centroid, var
    
    # input:
        # centorid: numpy.array
        # var: numpy.array
        # adv_data_normed: numpy.array, size[3,K]
    # output: 
        # adv_data_denormed: numpy.array, size[3,K]
    def denormalize(self, adv_data_normed, centroid, var):
        adv_data_denormed = adv_data_normed.transpose((1,0)) * var + centroid
        adv_data_denormed = adv_data_denormed.transpose((1,0))
        return adv_data_denormed

    # Remove nan value
    def removeNan(self, adv_data):
        return adv_data
    
    def get_attention_map(self, face_area, mask, out_root=None, y_p = None):
        kernel_size = int(face_area[2]/15)
        face_center_x = face_area[1] + face_area[3]/2. # row
        face_center_y = face_area[0] + face_area[2]/2. # col
        atten1_map = np.zeros((self.height,self.width), dtype =float)
        # local variance
        atten2_map = np.zeros((self.height,self.width), dtype = float)
        for i in range(face_area[1],int(face_area[1]+face_area[3])):
            for j in range(face_area[0], int(face_area[0]+face_area[2])):
                atten1_map[i,j] = math.exp(-0.01 * math.sqrt(pow(i-face_center_x,2) + pow(j-face_center_y,2)))
                atten2_map[i,j] = np.sum(mask[i-int(kernel_size/2):i+int((kernel_size+1)/2),j-int(kernel_size/2):j+int((kernel_size+1)/2)] , axis = (0,1))
                atten2_map[i,j] = 1/(atten2_map[i,j]+0.01)
        atten2_map[atten2_map>1] =  1        # background
        atten1_map = (atten1_map - np.min(atten1_map,(0,1)))/(np.max(atten2_map,(0,1)) - np.min(atten2_map,(0,1)))
        atten2_map = gaussBlur(atten2_map, 2, 10 , 10)
       
        atten_map = 0.5 * atten1_map + atten2_map
        return atten_map
        
    def localVar(self , img):
        # reference to matlab stdfilt(I) function
        img = img / 255.0
        h = np.ones((3,3))
        n = h.sum()
        
        c1 = cv2.filter2D(img**2, -1, h, borderType=cv2.BORDER_REFLECT) / n
        mean = cv2.filter2D(img, -1, h, borderType=cv2.BORDER_REFLECT)
        c2 = mean ** 2
        J = np.sqrt( np.maximum(c1-c2,0) )
        return J

            






# %% rebuild stripe 
# % load xp rebuild and xp origin. 
# %  We compare the two x_p. if x_p has not change at (i,j), then we donnot need to modify the projector image at (i,xp(i,j)))
import cv2
import numpy as np
import scipy.io as scio
import os
import shutil
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from math import exp,pow,sqrt
from Visualization import showXYZ, mesh, plot,  gaussBlur
import argparse
def load_txt(txtname):
    try:
        txt = np.loadtxt(txtname, delimiter=' ')
    except:
        txt = np.loadtxt(txtname, delimiter=',')
    return txt

def  Rebuild_stripe(x_p_rebuild, x_p):
    x_p = x_p.astype(int)
    W = 320
    H = 180
    x_p_rebuild = x_p_rebuild.astype(int)
    x_p_rebuild[x_p_rebuild<0] = 0
    x_p_rebuild[x_p_rebuild>W-1] = W-1 
    dataFolder = 'test_face_data//pictures//'
    imlist = []
    
    for idx in range(1,21):
        I_N_projector = cv2.imread(dataFolder+ 'project_grayloop//' + str(idx)+'.bmp',cv2.IMREAD_GRAYSCALE)
        I_N_projector = cv2.resize(I_N_projector,(W,H),interpolation = cv2.INTER_NEAREST)
        I_stripe_rebuild = I_N_projector
        for i in range(H):
            for j in range(W):
                if x_p_rebuild[i,j] > 0.1 :
                    I_stripe_rebuild[i][x_p[i][j]] = I_N_projector[i][x_p_rebuild[i][j]];
        imlist.append(I_stripe_rebuild)

    return imlist

def  Rebuild_phase(unnormalized_data):
    Xws = unnormalized_data[:,0]
    Yws = unnormalized_data[:,1]
    Zws = unnormalized_data[:,2]
    height = 180
    width = 320
    area = np.zeros((height, width),dtype=np.float)
    x_p = np.full((height, width),0.0)
    
    # load camera and projector parameters
    RTc = scio.loadmat('test_face_data/CamCalibResult.mat');
    Kc = RTc['KK']
    Rc_1 = RTc['Rc_1']
    Tc_1 = RTc['Tc_1']
    Kc = np.dot([[0.25,0,0],[0,0.25,0],[0,0,1]],Kc) #resize the image
    Ac = np.dot(Kc, np.append(Rc_1, Tc_1,axis = 1))
    print(Ac)
    print("Kc:",Kc)
    print("Rc",Rc_1)
    # load projector extric matrix
    RTp = scio.loadmat('test_face_data/PrjCalibResult.mat');
    Kp = RTp['KK']
    Rp_1 = RTp['Rc_1']
    Tp_1 = RTp['Tc_1']
    Kp = np.dot([[0.25,0,0],[0,0.25,0],[0,0,1]],Kp)
    Ap = np.dot(Kp, np.append(Rp_1, Tp_1,axis = 1))
        
    count_xy = 0 
    xcol = []
    ycol = []
        
    #read data and compute corresponding camera and projector cooridates
    for i in range(0,unnormalized_data.shape[0]):
        if np.isnan(Xws[i]) == False:
            count_xy = count_xy+1;
            point = np.array([Xws[i],Yws[i],Zws[i],1])
            usv = np.dot(Ap, point.T).astype(float) # the 
            usv_c = np.dot(Ac, point.T).astype(float)
            # vertical coordinate
            x = round(usv_c[1]/usv_c[2])
            # horizontal coordinate
            y = round(usv_c[0]/usv_c[2])
            xcol.append(x)
            ycol.append(y)
            x_p[x,y] = usv[0]/usv[2] # the horizontal coordinate in projector for (x,y) pixel in the captured image
            area[x,y] = 1.0
    
    xcol = np.array(xcol)
    ycol = np.array(ycol)
    face_area = np.array([np.min(ycol),np.min(xcol), np.max(ycol)-np.min(ycol), np.max(xcol)-np.min(xcol)],dtype=np.uint32)
    # print(face_area)
    # fig, ax = plt.subplots()
    # ax.imshow(area)
    # rect = patches.Rectangle((face_area[0],face_area[1]), face_area[2], face_area[3], linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    # fig.savefig("face_area.png") 
    
    # plot(x_p/width, "test_face_data/x_p.png") 
    # print(f"save original phase map at test_face_data/x_p.png")
    # showXYZ(unnormalized_data[:,0:3], f"test_face_data/original_before_rebuild.png", face_area)
    # print(f"save original 3D point cloud at test_face_data/original_before_rebuild.png")
    return x_p, face_area

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get adversarial illuminations')    
    parser.add_argument('--normal_pc', type=str, default="test_face_data/person1.txt")
    parser.add_argument('--adv_pc', type=str, default="test_face_data/adv_person1_untargeted_L1Loss_5.txt")
    parser.add_argument('--outfolder', type=str, default='test_face_data/person1/adversarial_fringe')        
    args = parser.parse_args()
    
    normal_pc = load_txt(args.normal_pc)
    x_p, _ = Rebuild_phase(normal_pc)
    
    adv_pc = load_txt(args.adv_pc)
    x_p_rebuild, _ = Rebuild_phase(adv_pc)
    
    OutFolder = args.outfolder
    imlist = Rebuild_stripe(x_p_rebuild, x_p)
    for idx in range(1,21):
        img = imlist[idx-1]
        cv2.imwrite(OutFolder+'//'+str(idx)+'.bmp', img)
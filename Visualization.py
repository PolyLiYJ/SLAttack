import pdb
import time
import random
import torch
import torch.optim as optim
import numpy as np
import scipy.io as scio
import matplotlib
from PIL import Image
import matplotlib.patches as patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
from scipy import signal
def plot(img, filename, face = [None]):
    if face[0] == None:
        # img = (img-np.min(img,axis=(0,1)))/(np.max(img, axis=(0,1))-np.min(img,axis=(0,1)))
        plt.imsave(filename, img)
    else:
        img = img[face[1]:face[1]+face[3],face[0]: face[0]+face[2]]
        # img = (img-np.min(img,axis=(0,1)))/(np.max(img, axis=(0,1))-np.min(img,axis=(0,1)))
        plt.imsave(filename, img)
    print(f"save imge at {filename}")

def mesh(face_area, pha, filename, zlabel = r'$z$'):
    [x, y, w, h]= face_area
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    yy=np.linspace(y,y+h+1,h)
    xx=np.linspace(x,x+w+1,w)
    fig = plt.figure()
    sub=fig.add_subplot(111,projection='3d')
    y_mesh,x_mesh=np.meshgrid(yy,xx,indexing='ij')
    surf=sub.plot_surface(y_mesh,x_mesh,pha[y:y+h,x:x+w],cmap=plt.cm.Purples)
    cb=fig.colorbar(surf,shrink=0.8,aspect=15,label='$z(x,y)$')

    sub.set_xlabel(r'$x$')
    sub.set_ylabel(r'$y$')
    sub.set_zlabel(zlabel)
    if filename == None:
        plt.savefig('mesh.png')
    else:
        plt.savefig(filename)
    plt.show()



def showXYZ(xyz, filename, face_area = None, axis = "off"):
    if face_area is not None:
        w = face_area[2]
        h = face_area[3]
        fig = plt.figure(figsize=(w*6/h,6))
    else:
        fig = plt.figure(figsize=(4.5,6))
    ax = fig.add_subplot(111, projection='3d')
    xs = xyz[:,0]
    ys = xyz[:,1]
    zs = xyz[:,2]
    ax.scatter(xs,ys,zs, marker=".",c='black', s=2)
    ax.view_init(100, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.axis(axis)
    plt.savefig(filename, bbox_inches='tight', pad_inches = 0)
    print("save fig at  " ,filename) 
    # plt.show()
    

def gaussBlur(img, sigma, H, W, _boundary='fill', _fillvalue=0):

    gaussKernel_x = cv2.getGaussianKernel(W, sigma, cv2.CV_64F)

    gaussKernel_x = np.transpose(gaussKernel_x)

    gaussBlur_x = signal.convolve2d(img, gaussKernel_x, mode="same",
                                    boundary=_boundary, fillvalue=_fillvalue)

    gaussKernel_y = cv2.getGaussianKernel(H, sigma, cv2.CV_64F)

    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode="same",
                                     boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy

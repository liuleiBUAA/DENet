import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import glob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import cv2
import os, shutil
import json

def gaussian_filter_density(gt):
    print gt.shape
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print 'generate density...'
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print 'done.'
    return density

root = '/home/liulei/Downloads/CSRNet-pytorch-master'

part_UCSD_train = os.path.join(root,'UCSD/train/images1')
part_UCSD_test = os.path.join(root,'UCSD/test/images1')
path_sets = [part_UCSD_train]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

data_test = []
for img_path in img_paths:
    data_test.append(img_path)

    gt_path = img_path.replace('.jpg', 'dots.png').replace('images1', 'ground_truth')
#     print (gt_path)
    img = cv2.imread(img_path,0)
    # img1=cv2.resize(img, (img.shape[1]*4,img.shape[0]*4),interpolation=cv2.INTER_LINEAR)
    k = cv2.imread(gt_path,0)
    k = k/np.max(k)
    k = k.astype(np.float)

    # mask_path = img_path.replace('images1', 'mask').replace('.jpg','BW.jpg')
    # mask = cv2.imread(mask_path,1)
    # ret, mask1 = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)
    # new_img = np.multiply(img, mask1 / 255)
    new_img=img
    # mask2 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)

    k = gaussian_filter(k, 3)
    # k = np.multiply(k, mask1 / 255)
    cv2.imwrite(img_path.replace('images1', 'images'), new_img)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images1', 'ground_truth'), 'w') as hf:
        hf['density'] = k


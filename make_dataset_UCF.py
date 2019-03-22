import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import glob
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
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print 'done.'
    return density


root = '/home/liulei/Downloads/CSRNet-pytorch-master'

data_test = []
img_paths = []
part_UCF = os.path.join(root, 'ORI_UCF')
path_sets = [part_UCF]
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    data_test.append(img_path)
    # creat data
    mask_path = img_path.replace('ORI_UCF', 'UCF_CC_50/mask')
    mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path.replace('.jpg', 'BW.jpg'), 1)
    ret, mask1 = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask1 = cv2.erode(mask1, kernel)

    new_img = np.multiply(img, mask1 / 255)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["annPoints"]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    GT = np.sum(k)
    str1=img_path.split('/ORI_UCF/')
    img_name=str1[1]
    file = open('./detection_result/UCF.txt', 'r')
    js = file.read()
    dict = json.loads(js)
    detection = dict[img_name]
    GT_detection = GT - detection


    cv2.imwrite(img_path.replace('ORI_UCF', 'UCF_CC_50'), new_img)
    mask2 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    k = gaussian_filter(k, 15)
    # k1 = np.multiply(k, mask2 / 255)
    # target_sum = np.sum(k1)
    #
    # k = gaussian_filter(k, 15)*GT_detection/target_sum
    k = np.multiply(k, mask2 / 255)

    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth').replace('ORI_UCF', 'UCF_CC_50'), 'w') as hf:
        hf['density'] = k

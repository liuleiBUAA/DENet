import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
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

root = '/home/liulei/Downloads/CSRNet-pytorch-master/'

# part_A_train = os.path.join(root,'Shanghai/newpartA/train_data','images')
# part_A_test = os.path.join(root,'Shanghai/newpartA/test_data','images')
# part_B_train = os.path.join(root,'Shanghai/newpartB/train_data','images')
# part_B_test = os.path.join(root,'Shanghai/newpartB/test_data','images')
part_A_train = os.path.join(root,'Shanghai/part_A_final/train_data','images')
part_A_test = os.path.join(root,'Shanghai/part_A_final/test_data','images')
part_B_train = os.path.join(root,'Shanghai/part_B_final/train_data','images')
part_B_test = os.path.join(root,'Shanghai/part_B_final/test_data','images')

path_sets = [part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print img_path
    mask_path = img_path.replace('images', 'images/mask').replace('part_B_final','newpartB')
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    img = cv2.imread(img_path)

    mask = cv2.imread(mask_path.replace('.jpg', 'BW.jpg'), 1)
    ret, mask1 = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)
    new_img = np.multiply(img, mask1 / 255)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1

    GT = np.sum(k)
    str1 = img_path.split('/images/')
    img_name = str1[1]
    file = open('./detection_result/testB.txt', 'r')
    js = file.read()
    dict = json.loads(js)
    detection = dict[img_name]
    GT_detection = GT - detection

    k = gaussian_filter(k,15)
    new_img = np.multiply(img, mask1 / 255)
    cv2.imwrite(img_path.replace('part_B_final','newpartB'),new_img)
    mask2 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)

    k1 = np.multiply(k, mask2 / 255)
    target_sum = np.sum(k1)

    k = gaussian_filter(k, 15) * GT_detection / target_sum
    k = np.multiply(k, mask2 / 255)

    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth').replace('part_B_final','newpartB'), 'w') as hf:
            hf['density'] = k


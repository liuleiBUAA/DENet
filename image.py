import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    scale=1
    scale_out=1
    real_scale=scale/scale_out
    change=np.square(scale / scale_out)

    # finish
    if True:
        crop_size = (img.size[0] / 8 * 8, img.size[1] / 8 * 8)
        # dx = int(random.random() * img.size[0] * 1 / 2)
        # dy = int(random.random() * img.size[1] * 1 / 2)
        dx=0
        dy=0
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        target = cv2.resize(target, (target.shape[1] / real_scale, target.shape[0] / real_scale),
                            interpolation=cv2.INTER_CUBIC) * change
        #
        # print np.sum(target)
        # target = cv2.resize(target, (target.shape[1] / scale, target.shape[0] / scale), interpolation=cv2.INTER_CUBIC)
        # target = cv2.resize(target, (target.shape[1] * scale_out, target.shape[0] * scale_out), interpolation=cv2.INTER_CUBIC)
        # target = target*change
        return img,target



#
    # target = cv2.resize(target, (target.shape[1] / scale, target.shape[0] / scale), interpolation=cv2.INTER_CUBIC)
    # target = cv2.resize(target, (target.shape[1] * scale_out, target.shape[0] * scale_out), interpolation=cv2.INTER_CUBIC)
    # target = target * change
    # target = cv2.resize(target, (target.shape[1] / 8, target.shape[0] / 8), interpolation=cv2.INTER_CUBIC) * 64
    # target = cv2.resize(target, (target.shape[1] / 4, target.shape[0] / 4), interpolation=cv2.INTER_CUBIC) * 16
    # target = cv2.resize(target, (target.shape[1] / 2, target.shape[0] / 2), interpolation=cv2.INTER_CUBIC) * 4
#
#
    return img, target
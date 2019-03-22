import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from matplotlib import pyplot as plt
from image import *
from model import CSRNet, SANet, MCNN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
root = './Shanghai/'
#now generate the ShanghaiA's ground truth
# part_A_train = os.path.join(root,'part_A_final/train_data','images')
# part_A_test = os.path.join(root,'part_A_final/test_data','images')
# part_B_train = os.path.join(root,'part_B_final/train_data','images')
# part_B_test = os.path.join(root,'part_B_final/test_data','images')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'newpartA/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'newpartB/test_data','images')
part_1_test = os.path.join(root,'test/200608','images')
UCF1=os.path.join(root,'newpartB/test_data','images')
UCSD_test='./UCSD/test/images1'
path_sets = [part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

model = CSRNet()

model = model.cuda()
checkpoint = torch.load('./pre-trianed model/shanghaiBmodel_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
mae = 0
mse = 0
# global args
# testUCF= './json/UCF_data0_test.json'
# with open(testUCF, 'r') as outfile:
#     val_list = json.load(outfile)

# def validate(val_list, model):
#     print ('begin test')
#     test_loader = torch.utils.data.DataLoader(
#         dataset.listDataset(val_list,
#                             shuffle=False,
#                             transform=transforms.Compose([
#                                 transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                             std=[0.229, 0.224, 0.225]),
#                             ]), train=False),
#         batch_size=1)
#
#     model.eval()
#
#     mae = 0
#     mse = 0
#
#     for i, (img, target, GT_detection, target_sum) in enumerate(test_loader):
#         img = img.cuda()
#         img = Variable(img)
#         output = model(img)
#
#         GT_detection = GT_detection.type(torch.FloatTensor).unsqueeze(0).cuda()
#         GT_detection = Variable(GT_detection)
#         # a=output.detach().cpu().numpy()
#         # b=a.shape
#         # a = a.reshape(b[2], b[3])
#         # plt.imsave('result', a, cmap='jet')
#         mae += abs(output.data.sum() - GT_detection.data.sum())
#         # mae += abs(output.detach().cpu().sum().numpy()-GT_detection.data.numpy())
#         # mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
#         mse += (output.data.sum() - GT_detection.data.sum()) * (output.data.sum() - GT_detection.data.sum())
#         # mse += np.square(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
#     mae = mae / len(test_loader)
#     mse = np.sqrt(mse / len(test_loader))
#     print(' * MAE {mae:.3f} '
#           .format(mae=mae))
#     print(' * MSE {mse:.3f} '
#           .format(mse=mse))
#
#     return mae

# prec1 = validate(val_list, model)
#
file=open('./detection_result/testB.txt','r')
js=file.read()
dict=json.loads(js)
GTresult=[]
Eresult=[]
for i in xrange(len(img_paths)):

    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()

    # gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth').replace('newpartA/test_data','part_A_final/test_data'),'r')
    gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth').replace('newpartB/test_data','part_B_final/test_data'),'r')

    # gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth'),'r')

    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    # # tensor to numpy

    str1=img_paths[i].split('/images/')

    img_name=str1[1]
    strname = './result/' + img_name


    a = output.detach().cpu().numpy()
    b = a.shape
    a = a.reshape(b[2], b[3])
    img1 = cv2.resize(a, (a.shape[1] * 8, a.shape[0] * 8), interpolation=cv2.INTER_LINEAR)
    plt.imsave(strname, img1, cmap='jet')



    estimated=output.detach().cpu().sum().numpy()+dict[img_name]
    GT=np.sum(groundtruth)

    # draw = ImageDraw.Draw(a)
    # draw.text((0, 0), "Hello", fill=(255, 0, 0))
    mae += abs(estimated - np.sum(groundtruth))
    mse += np.square(estimated - np.sum(groundtruth))
    # mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    # mse += (output.detach().cpu().sum().numpy() - np.sum(groundtruth)) * (output.detach().cpu().sum().numpy() - np.sum(groundtruth))

    GTresult.append(GT)
    Eresult.append(estimated)
    # output=[]
    # print i,mae,mse
    if abs(GT-estimated)<25:
        print i,GT,estimated
print mae/len(img_paths), np.sqrt(mse/len(img_paths))

t = np.arange(len(img_paths))
plt.figure()
plt.plot(t, GTresult, 'r-', t, Eresult, 'b-')
plt.show()
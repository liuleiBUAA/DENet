import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
import coco
import utils
import model as modellib
import visualize
import scipy.io as io
import cv2
# matplotlib inline

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'boos.walk(path)wl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
# file_names=os.listdir(IMAGE_DIR)
# file_names.sort(key= lambda x:int(x[4:-4]))
dict={}

for file_name in file_names:
    # print(type(file_names))
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name),as_grey=False)
    a=len(image.shape)
    if len(image.shape)==2:
        image1=skimage.color.gray2rgb(image)
        image=image1
# Run detection
    results = model.detect([image], verbose=1)

# Visualize results
    r = results[0]
    a=len(r['scores'])
    strname=file_name
    number_of_person=visualize.display_instances(strname, image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
    number = number_of_person
    index = file_name
    dict[index] = int(number)
    os.remove(os.path.join(IMAGE_DIR, file_name))

file=open('trainB.txt','w')
a=json.dump(dict,file)
# file.write(aaa)
file.close()
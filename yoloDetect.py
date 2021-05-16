import os
import sys
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from skimage.segmentation import active_contour
from skimage.filters import gaussian
from cv2 import equalizeHist
from PIL import Image, ImageDraw
from torchvision import transforms


spine_pos_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']

'''
read yolo label
label_path: path of label filename.txt with yolo form [class, normalized x center, normalized y center, normalized width, normalized height, confidence]
img_size: size of full image (used to recover from normalized coordinate)
poss: positions list, if S is detected count back the position from S, else count back from L5

return: a list of dictionary with [label, x_center , y_center, box_width, box_height, conf, path]
sorted by y in ascending order.
'''
def read_label(label_path, img_size, conf_thresh = 0.8, poss = spine_pos_list):

    img_width, img_height = img_size

    with open(label_path, mode='r') as f:
        lines = f.readlines()
    
    temp_list = []
    for line in reversed(lines):

        temp_dict = dict()

        # read yolov5 output format
        lst = [float(a) for a in line.strip('\n').split(' ')]
        label, x_center_norm, y_center_norm, width_norm, height_norm, predict_conf = lst

        if predict_conf < conf_thresh:
            continue

        x_center = int(x_center_norm * img_width)
        y_center = int(y_center_norm * img_height)
        box_width = int(width_norm * img_width)
        box_height = int(height_norm * img_height)

        temp_dict['label'] = 'S' if label == 0 else 'others'
        temp_dict['x_center'] = x_center
        temp_dict['y_center'] = y_center
        temp_dict['box_width'] = box_width
        temp_dict['box_height'] = box_height
        temp_dict['conf'] = predict_conf
        temp_dict['path'] = label_path

        temp_list.append(temp_dict)

    temp_list.sort(key=lambda x: x.get('y_center'))

    # if find Sacral in image
    if temp_list[-1]['label'] == 'S':
        for idx in range(len(temp_list)-1):
            temp_list[-(idx+2)]['label'] = poss[-(idx+1)]
    # if no Sacral, directly set the last vertext to L5
    else:
        for idx, temp in enumerate(reversed(temp_list)):
            temp['label'] = poss[-(idx+1)]


    return temp_list

'''
seg_by_label
img_path: an image of complete X-rays
label_path: path of a file with standard yolov5 format
include_S: if True the return value will include S else not

return three list
[image of single vertex], [center coordinate], [position]
'''

def seg_by_label(img_path, label_path, include_S=False):

    seg_imgs = []
    boxs = []
    positions = []

    full_img = Image.open(img_path)
    labels = read_label(label_path, full_img.size)
    for label in labels:

        if label['label'] == 'S' and not include_S:
            continue

        xc, yc = label['x_center'], label['y_center']
        w, h = label['box_width'], label['box_height']
        seg_imgs.append(full_img.crop((xc-w//2, yc-h//2, xc+w//2, yc+h//2)))
        boxs.append((xc-w/2, yc-h/2, xc+w/2, yc+h/2))
        positions.append(label['label'])

    return seg_imgs, boxs, positions

def executeYolo(src, dest, name, conf=0.7):

    args = [
        'C:/Users/user/OneDrive/桌面/ressult_good/runs_result/exp/weights/best.pt', # weight path
        conf, # conf
        src, # source path
        dest, # dest path
        name, # filename
    ]

    os.chdir('./content/yolov5/')
    os.system('python ./detect.py --weights {} --img 400 --conf {} --source {} --save-conf --save-txt --project {} --name {}'.format(*args))
    os.chdir('./../../')

    
    
"""data

Keys
---------
filename: str
    filename of image (.bmp)
filepath: str
    absolute path of image file
x_label: [[int]]
    n labels in one image with m points(x) for each label
y_label: [[int]]
    n labels in one image with m points(y) for each label
type: str
    type of vertexs => normal, unsure, compre, burst
position: str
    position of vertexs => S, L5, L4..., T11, T10,...

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def MyLoadDATA(path='.\\content\\DATA'):

    """
    load DATA according to the json file with VGG annotator format(https://www.robots.ox.ac.uk/~vgg/software/via/)
    transfer to my data format which preserve only important information
    """

    label_files = [os.path.join(path, DATA_dir, lablel_file) for DATA_dir in os.listdir(path) for lablel_file in os.listdir(os.path.join(path, DATA_dir)) if lablel_file.endswith('.json')]

    data = list()

    for label_file in label_files:
        with open(label_file, mode='r', encoding='utf-8') as f:
            print(label_file)
            load_json = json.load(f)

            for key in load_json['_via_img_metadata']:
                
                this_img_data = load_json['_via_img_metadata'][key]

                temp_dict = dict()
                temp_dict['filename'] = this_img_data['filename']
                temp_dict['filepath'] = os.path.join(path, this_img_data['filename'].split('_')[0], this_img_data['filename'])
                temp_dict['x_label'] = [k['shape_attributes']['all_points_x'] for k in this_img_data['regions']] # [[12], [12], ...]
                temp_dict['y_label'] = [k['shape_attributes']['all_points_y'] for k in this_img_data['regions']] # [[12], [12], ...]
                temp_dict['position'] = [k['region_attributes']['position'] for k in this_img_data['regions']]   # ['L1', 'L2', ...]
                temp_dict['type'] = [k['region_attributes']['type'] for k in this_img_data['regions']]           # ['normal', compre, ...] 

                data.append(temp_dict)

    return data

def plotLabel(data):

    """
    plot the data according to the original shape
    """

    img = Image.open(data['filepath'])
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')

    get_line_style = {
    'normal': 'y-',
    'compre': 'r-',
    'burst' : 'b-',
    'unsure': 'g-',
    }

    for idx in range(len(data['x_label'])):

        line_style =  get_line_style[data['type'][idx]]

        for idx_dot in range(len(data['x_label'][idx])):
            x1 = data['x_label'][idx][idx_dot]
            y1 = data['y_label'][idx][idx_dot]
            x2 = data['x_label'][idx][idx_dot + 1 if idx_dot + 1 < len(data['x_label'][idx]) else 0]
            y2 = data['y_label'][idx][idx_dot + 1 if idx_dot + 1 < len(data['x_label'][idx]) else 0]
            # print([x1, y1], [x2, y2])
            plt.plot([x1, x2], [y1, y2], line_style, linewidth = 1)
        
        if data['position'][idx] != 'S':
            x_mid = sum(data['x_label'][idx])/len(data['x_label'][idx])
            y_mid = sum(data['y_label'][idx])/len(data['y_label'][idx])
            plt.plot(x_mid, y_mid, 'r+')

    plt.show()


def plotBox(data, width = 35, height = 35):

    img = Image.open(data['filepath'])
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')

    boxs = []
    mids = []

    get_line_style = {
    'normal': 'y-',
    'compre': 'r-',
    'burst' : 'b-',
    'unsure': 'g-',
    }

    for idx in range(len(data['x_label'])):

        line_style =  get_line_style[data['type'][idx]]

        x_max = min(max(data['x_label'][idx]) + width, img.size[0])
        x_min = max(min(data['x_label'][idx]) - width, 0)
        y_max = min(max(data['y_label'][idx]) + height, img.size[1])
        y_min = max(min(data['y_label'][idx]) - height, 0)

        plt.plot([x_max, x_max],[y_max, y_min], line_style, linewidth = 1)
        plt.plot([x_min, x_min],[y_max, y_min], line_style, linewidth = 1)
        plt.plot([x_max, x_min],[y_max, y_max], line_style, linewidth = 1)
        plt.plot([x_max, x_min],[y_min, y_min], line_style, linewidth = 1)

        # store the center point (except for S)
        if data['position'][idx] != 'S':

            x_mid = sum(data['x_label'][idx])/len(data['x_label'][idx])
            y_mid = sum(data['y_label'][idx])/len(data['y_label'][idx])

            plt.plot(x_mid, y_mid, 'r+')

            boxs.append((x_min, y_min, x_max, y_max))
            mids.append((x_mid, y_mid))

    plt.show()


def plotBoxZoom(data, width=35, height=35):

    img = Image.open(data['filepath'])
    plt.figure(figsize = (10, 10))

    boxs = []
    mids = []

    get_dot_style = {
    'normal': 'y+',
    'compre': 'r+',
    'burst' : 'b+',
    'unsure': 'g+',
    }

    for idx in range(len(data['x_label'])):

        x_max = min(max(data['x_label'][idx]) + width, img.size[0])
        x_min = max(min(data['x_label'][idx]) - width, 0)
        y_max = min(max(data['y_label'][idx]) + height, img.size[1])
        y_min = max(min(data['y_label'][idx]) - height, 0)


        if data['position'][idx] != 'S':

            x_mid = sum(data['x_label'][idx])/len(data['x_label'][idx])
            y_mid = sum(data['y_label'][idx])/len(data['y_label'][idx])

            plt.plot(x_mid, y_mid, 'r+')

            boxs.append((x_min, y_min, x_max, y_max))
            mids.append((x_mid, y_mid))

    nrows = int(len(boxs)/3)+1
    ncols = 3

    for idx, box in enumerate(boxs):

        # subplot index start from 1
        plt.subplot(nrows, ncols, idx+1)
        plt.imshow(img.crop(box), cmap='gray')
        mid = ( mids[idx][0]-box[0], mids[idx][1]-box[1] )
        # unpack mid
        plt.plot(*mid, get_dot_style[data['type'][idx]], markersize=15)


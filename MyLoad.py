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
from cv2 import equalizeHist
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from segfunct import edge_segment, get_corner
from yoloDetect import read_label


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

def plotBox(data, width = 20, height = 20):

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

def plotBoxZoom(data, width=20, height=20):

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


def MyToSegmentDATA(data, dest='./content/VsegDATA', histeq=False):

    """
    take the output of MyLoadDATA as input
    save each box of image to /content/segDATA with structure:

    /content/VsegDATA/
        /image/
            /normal/
                [id]_[FILEn]_[position].bmp
            /compre/
                [id]_[FILEn]_[position].bmp
            /burst/
                [id]_[FILEn]_[position].bmp
            /unsure/
                [id]_[FILEn]_[position].bmp
        /mask/
            /normal/
                [id]_[FILEn]_[position].txt
            /compre/
                ...
        /label.json
    """

    idx = 1
    if os.path.isdir(dest):
        while os.path.isdir(dest + str(idx)):
            idx += 1
        dest = dest + str(idx)
    
    os.mkdir(dest)

    excess = 20

    label_dict = dict()
    label_file_path = os.path.join(dest, 'label.json')

    path_selector = dict()
    cat_list = ['normal', 'compre', 'burst', 'unsure']
    img_dirs = [os.path.join(dest, 'image', c) for c in cat_list]
    msk_dirs = [os.path.join(dest, 'mask', c) for c in cat_list]
    for cat, img_dir, msk_dir in zip(cat_list, img_dirs, msk_dirs):
        path_selector[cat] = (img_dir, msk_dir)
        os.makedirs(img_dir)
        os.makedirs(msk_dir)
        label_dict[cat] = dict()

    for d in data:
        
        img = Image.open(d['filepath'])
        img_width, img_height = img.size
        img_filename = d['filename'].split('.')[0]

        for idx in range(len(d['x_label'])):

            # pass all S
            if d['position'][idx] == 'S':
                continue

            # check all corp image are in correct range
            x_min = max(0, min(d['x_label'][idx]) - excess)
            x_max = min(img_width, max(d['x_label'][idx]) + excess)
            y_min = max(0, min(d['y_label'][idx]) - excess)
            y_max = min(img_height, max(d['y_label'][idx]) + excess)

            # save the new coordinate x,y after crop the image
            # (left top as (0,0))
            temp_dict = dict()
            temp_dict['xs'] = [k - x_min for k in d['x_label'][idx]]
            temp_dict['ys'] = [k - y_min for k in d['y_label'][idx]]

            box = (x_min, y_min, x_max, y_max)

            seg_img = img.crop(box)
            seg_img_filename = img_filename + '_{}.bmp'.format(d['position'][idx])
            seg_mask_filename = img_filename + '_{}.txt'.format(d['position'][idx])

            # draw mask
            mask = Image.new('1', seg_img.size)
            ImageDraw.Draw(mask).polygon([(x,y) for x, y in zip(temp_dict['xs'], temp_dict['ys'])], outline=1, fill=1)
            mask = np.array(mask)
            
            # ex: label_dict[__filename__][xs] = [0, 4, 5, ....]
            #     label_dict[__filename__][ys] = [2, 5, 7, ....] 
            label_dict[d['type'][idx]][seg_img_filename] = temp_dict

            seg_img_dirpath, seg_mask_dirpath = path_selector[d['type'][idx]]

            seg_img_filepath = os.path.join(seg_img_dirpath, seg_img_filename)
            # do histogram equalization
            if histeq == True:
                seg_img_array = np.array(seg_img)
                seg_img_array = equalizeHist(seg_img_array)
                seg_img = Image.fromarray(seg_img_array)
            seg_img.save(seg_img_filepath)

            seg_mask_filepath = os.path.join(seg_mask_dirpath, seg_mask_filename)
            np.savetxt(seg_mask_filepath, mask, fmt='%d')

    with open(label_file_path, mode='w', encoding='utf-8') as f:
        json.dump(label_dict, f)

def MyResultVisualize(in_dir, result_dir, out_dir=None, conf_thresh=0.8):

    if out_dir == None:
        out_dir = os.path.join(result_dir, 'visual')

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # fid format [id]_FILE[n]: 1234568_FILE0
    fids = [fname.split('.')[0] for fname in os.listdir(in_dir) if fname.endswith('.bmp')]
    img_paths = [os.path.join(in_dir, fname) for fname in os.listdir(in_dir) if fname.endswith('.bmp')]
    
    label_dir = os.path.join(result_dir, 'labels')
    msk_dir = os.path.join(result_dir, 'masks')
    label_paths = [os.path.join(label_dir, fid+'.txt') for fid in fids]

    for label_path, img_path, fid in zip(label_paths, img_paths, fids):

        img = Image.open(img_path)
        label = read_label(label_path, img.size)

        outimg = np.zeros(img.size[::-1])
        ref_points = []

        # print(img.size)
        plt.figure(figsize=(20,10))

        for l in label:

            msk_path = os.path.join(msk_dir, fid+'_'+l['label']+'.txt')
            
            if not os.path.isfile(msk_path):
                print('warning: file not found {}'.format(msk_path))
                continue
            elif 'S' in msk_path:
                continue

            msk = np.loadtxt(msk_path)
            xc, yc, w, h = l['x_center'], l['y_center'], l['box_width'], l['box_height']

            for i in range(msk.shape[0]):
                for j in range(msk.shape[1]):
                    outimg[i + (yc-h//2) , j + (xc-w//2)] = msk[i,j]

            seg_n = 7

            corners = get_corner(mask=msk, rotate=True)
            a = edge_segment(corners[0], corners[2], msk, n=seg_n)
            b = edge_segment(corners[1], corners[3], msk, upper=False, n=seg_n)
            for i in range(len(a)):
                a[i], b[i] = (a[i][0]+(xc-w//2), a[i][1]+(yc-h//2)), (b[i][0]+(xc-w//2), b[i][1]+(yc-h//2))
            ref_points.extend([a,b])

        outimg = np.ma.masked_where(outimg==0, outimg)
        plt.imshow(img, cmap='gray')
        plt.imshow(255*outimg, cmap='jet', alpha=0.5)

        for idx in range(0, len(ref_points), 2):
            for upoint, lpoint in zip(ref_points[idx], ref_points[idx+1]):
                plt.plot(*upoint, 'r.')
                plt.plot(*lpoint, 'r.')
                plt.plot([upoint[0], lpoint[0]], [upoint[1], lpoint[1]], 'y-')

        # for points in ref_points:
        #     for point in points:
        #         plt.plot(*point, 'r.')

        plt.axis('off')
        plt.savefig(os.path.join(out_dir, fid+'.jpg') ,bbox_inches='tight')




if __name__ == '__main__':

    # MyToSegmentDATA(MyLoadDATA(), dest='./content/VHsegDATA', histeq=True)
    in_dir = r'C:\Users\user\OneDrive\桌面\vertex\content\DATA_unlabel\07227002'
    result_dir = r'C:\Users\user\OneDrive\桌面\vertex\content\pred_result\07227002'
    MyResultVisualize(in_dir, result_dir)
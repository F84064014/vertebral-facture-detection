import os
import sys
import cv2
import time
from numpy.lib.function_base import average
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

'''
given two points and the mask file, cut the line between two points into n equally long lines,
then fit each cut point onto the mask

cord1: points1(x1,y1)
cord2: points2(x2,y2)
outline: mask(PIL image or numpy array)
upper: is the upper vertex or lower vertex
n: the number of segmentation
'''
def edge_segment(cord1, cord2, outline, angle=0, upper=True, n=3):

    if cord1[0] < cord2[0]:
        cord1, cord2 = cord2, cord1

    if isinstance(outline, Image.Image):
        TF.rotate(outline, angle, expand=True)
        outline = np.array(outline)
    else:
        outline = np.array(TF.rotate(Image.fromarray(outline), angle, expand=True))
    
    kx = [int(cord2[0] + i*(cord1[0]-cord2[0])/n) for i in range(n+1)]
    ky = [int(cord2[1] + i*(cord1[1]-cord2[1])/n) for i in range(n+1)]

    orthogonal_vec = np.array([-(cord1[1]-cord2[1]), cord1[0]-cord2[0]])
    orthogonal_vec = orthogonal_vec / np.sqrt(np.square(orthogonal_vec[0]) + np.square(orthogonal_vec[1]))

    if upper == False:
        orthogonal_vec *= -1

    intersection = []

    for i in range(1, len(kx)-1):
        t=0
        if outline[ky[i], kx[i]] == True:
            i1, i2 = kx[i], ky[i]
            while True:
                t1 = round(kx[i] - t * orthogonal_vec[0])
                t2 = round(ky[i] - t * orthogonal_vec[1])
                if t1<0 or t2<0 or t1>=outline.shape[1] or t2>=outline.shape[0] or outline[t2, t1] == False:
                    intersection.append((i1,i2))
                    break
                else:
                    t+=1
                    i1, i2 = t1, t2
        else:
            i1, i2 = kx[i], ky[i]
            while True:
                t1 = round(kx[i] + t * orthogonal_vec[0])
                t2 = round(ky[i] + t * orthogonal_vec[1])
                if t1<0 or t2<0 or t1>=outline.shape[1] or t2>=outline.shape[0] or outline[t2, t1] != False:
                    intersection.append((t1,t2))
                    break
                else:
                    t+=1
                    i1, i2 = t1, t2

    intersection.append((kx[-1], ky[-1]))
    intersection = [(kx[0], ky[0])] + intersection

    return intersection

'''
get the four corners by the mask
mask: mask data (numpy txt)
rotate: if True, find the corner by the 45deg rotated image

find the corner of mask by finding the first occurence point of mask from left, right, top and bottom
'''
# get four corner of of an vertex image
def get_corner(mask, rotate=False):

    outline = mask

    corners = []

    if rotate == True:
        temp = TF.rotate(Image.fromarray(outline), angle=45, expand=True)
        outline = np.array(temp)

    for i in range(outline.shape[0]):
        if np.max(outline[i, :]) != 0:

            max_idx = [k[0] for k in np.argwhere(outline[i, :] == np.max(outline[i, :]))]
            corners.append((max_idx[int(len(max_idx)/2)], i))
            break

    for i in range(outline.shape[0]-1, 0, -1):
        if np.max(outline[i, :]) != 0:

            max_idx = [k[0] for k in np.argwhere(outline[i, :] == np.max(outline[i, :]))]
            corners.append((max_idx[int(len(max_idx)/2)], i))
            break

    for i in range(outline.shape[1]):
        if np.max(outline[:, i]) != 0:

            max_idx = [k[0] for k in np.argwhere(outline[:, i] == np.max(outline[:, i]))]
            corners.append((i, max_idx[int(len(max_idx)/2)]))
            break

    for i in range(outline.shape[1]-1, 0, -1):
        if np.max(outline[:, i]) != 0:

            max_idx = [k[0] for k in np.argwhere(outline[:, i] == np.max(outline[:, i]))]
            corners.append((i, max_idx[int(len(max_idx)/2)]))
            break

    if rotate == True:
        r_corner = []
        rad45 = -np.deg2rad(45)
        rot = np.array([[np.cos(rad45), -np.sin(rad45)],[np.sin(rad45), np.cos(rad45)]])

        for idx, corner in enumerate(corners):
            k = np.array( [corner[0]-temp.size[0]/2, corner[1]-temp.size[1]/2])
            corner = np.matmul(k, rot)
            corner = (corner[0]+mask.shape[1]/2, corner[1]+mask.shape[0]/2)
            corners[idx] = corner

    if len(corners) != 4:
        return corners

    corners.sort(key=lambda c: c[0], reverse=False)
    if corners[0][1] > corners[1][1]:
        corners[1], corners[0] = corners[0], corners[1]
    if corners[2][1] > corners[3][1]:
        corners[2], corners[3] = corners[3], corners[2]
        

    return corners

'''
get the four corners by the mask
mask: mask data (numpy txt)

find the corner of the mask by finding the minimize the function:
    for corner in [corner1 to corner4]
        corner = argmin (distance to corner)**2 + (distance to left/right boarder)**2

look like it works well but no theoratical proof
'''

def minCornerDistance(input_mask, angle=0):
    
    # do rotation
    mask = np.array(TF.rotate(Image.fromarray(input_mask), angle, expand=True))
    
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # several cluster of contours may be found
    if len(contours) > 1:
        contours.sort(key=lambda x: x.shape[0], reverse=True)
        contours = contours[0]

    idxs = []

    conts = np.squeeze(np.array(contours))
    upper_left = np.square(conts[:,0]) + np.square(conts[:,1])#+np.square(conts[:,0])
    idxs.append(np.argmin(upper_left))

    upper_right = np.square(conts[:,0]-mask.shape[1]) + np.square(conts[:,1])#-np.square(conts[:,0])
    idxs.append(np.argmin(upper_right))

    lower_right = np.square(conts[:,0]-mask.shape[1]) + np.square(conts[:,1]-mask.shape[0])#-np.square(conts[:,0])
    idxs.append(np.argmin(lower_right))

    lower_left = np.square(conts[:,0]) + np.square(conts[:,1]-mask.shape[0])#+np.square(conts[:,0])
    idxs.append(np.argmin(lower_left))

    # return [conts[idx] for idx in idxs]
    return conts[idxs]

'''
get the angle of each vertext by yolo result (center point)

boxs: list of box (left, top, right, bot)
'''

def get_angle(boxs):

    centers = []
    for box in boxs:
        xc = (box[0] + box[2]) / 2
        yc = (box[1] + box[3]) / 2
        centers.append((xc, yc))
    
    for i in range(len(centers)):
        
        slope_last, slope_next = None, None

        if i != 0:
            slope_last = (centers[i-1][0] - centers[i-1][0]) / (centers[i][1] - centers[i][1])
        slope_next = (centers[i][0] - centers[i][0]) / (centers[i+1][1] - centers[i+1][1])


if __name__ == '__main__':

    print('this is demo code')
    for i in range(10):
        binary = np.loadtxt(os.path.join('testmasked', 'test{}.txt'.format((i))), dtype=np.uint8)
        cs = minCornerDistance(binary)

        plt.figure()
        plt.imshow(binary)
        
        # cs = np.array(cs).squeeze()

        for c in cs:
            print(c)
            plt.plot(*c, 'r+')

        plt.show()
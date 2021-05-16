import os
import pickle
import numpy as np
from scipy.ndimage.fourier import _get_output_fourier

import torchvision.transforms.functional as TF

from PIL import Image
from segfunct import minCornerDistance, edge_segment

def get_fmodel(path):

    with open(path, 'rb') as f:
        md = pickle.load(f)

    if md != None:
        print('successfully load model of type {}'.format(type(md)))
        return md
    else:
        raise ValueError('Empty model found at {}'.format(path))

'''
masks: input a list of binary mask
positions: input a list of position of binray mask
md: input model
angle: rotation. rotate to the central line could lead to well beeter result
'''
def fracturePredict(mask, position, mds, angle=0, outfp=None):

    # for mask, position in zip(masks, positions):

    cs = minCornerDistance(mask, angle=angle)

    us = edge_segment(cs[0], cs[1], mask, angle=angle, n=6)
    ds = edge_segment(cs[2], cs[3], mask, angle=angle, upper=False, n=6)

    feature = np.sqrt(np.sum(np.square(np.array(us) - np.array(ds)), axis=1))

    # Len0, Len3, Len6
    feature = [feature[i] for i in [0, 3, 6]]
    # ratio03, ratio06, ratio36
    feature += [feature[1]/feature[0], feature[2]/feature[0], feature[2]/feature[1]]
    # pos_encode
    feature += [1 if 'L' in position else 0]

    if outfp != None:
        if not os.path.isfile(outfp):
            with open(outfp, 'w') as outf:
                outf.write('{}\n'.format('Len0 Len3 Len6 ratio03 ratio06 ratio36 pos_encode'))
                outf.write('{}\n'.format(' '.join([str(f) for f in feature])))
        else:
            with open(outfp, 'a') as outf:
                outf.write('{}\n'.format(' '.join([str(f) for f in feature])))

    # multiple prediction but here input only one vertex
    preds = []
    for i in range(len(mds)):
        preds.append(mds[i]['dec'].predict([feature]))

    preds = np.where(np.sum(np.array(preds).T, axis=1) > 3, 1, 0)

    return preds[0]
import os
import copy
import numpy as np
from PIL import Image
from matplotlib import lines, pyplot as plt
from torch._C import dtype

from yoloDetect import read_label
from segfunct import minCornerDistance


'''
img: input full image
masks: list of prediction result for segmentation
conds: list of integer represent the cnodition of vertex
'''
def ResultVisiualize(img_fp, msks, conds, boxs, dest_fp=None):
    
    img = Image.open(img_fp)
    plt.imshow(img, cmap='gray')
    plt.gca().set_axis_off()
    
    for msk, box, cond in zip(msks, boxs, conds):

        # new image is outimg with padding = 0
        new_img = np.zeros(np.asarray(img).shape)

        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                new_img[int(box[1]+i),int(box[0]+j)] = msk[i,j]

        # masked = np.ma.masked_where(new_img < 0.8 , new_img)
        masked = np.where(new_img > 0.8, 1, np.nan)
        col = 'jet' if cond==1 else 'autumn'

        plt.imshow(masked, alpha=0.5, cmap=col)
        # plt.text(box[0], box[1], 'this is good')

        cs = minCornerDistance(msk, 0)

    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if dest_fp != None:
        plt.savefig(dest_fp, bbox_inches = 'tight', pad_inches = 0)
    else:
       plt.show()


if __name__=='__main__':

    print('hi this is demo code')

    ifp = r'C:\Users\user\OneDrive\桌面\vertex\content\DATA\16060746\16060746_FILE0.bmp'
    msk_root = r'C:\Users\user\OneDrive\桌面\vertex\testResult\16060746\masks'
    yolo_root = r'C:\Users\user\OneDrive\桌面\vertex\testResult\16060746\labels\16060746_FILE0.txt'

    names = [
        os.path.join(msk_root, fn) for fn in ['16060746_FILE0L1.txt', '16060746_FILE0L2.txt', '16060746_FILE0L3.txt', '16060746_FILE0L4.txt', '16060746_FILE0L4.txt']
    ][::-1]

    msks = []

    for idx, name in enumerate(names):
        msks.append(np.loadtxt(name, dtype=np.int32))


    data = read_label(yolo_root, Image.open(ifp).size)
    coord = [(int(data[i]['x_center']-data[i]['box_width']//2), int(data[i]['y_center']-data[i]['box_height']//2)) for i in range(-2, -8, -1) if data[i]['conf'] > 0.5]
    # coord = [(data[i]['x_center']-data[i]['box_width'], data[i]['y_center']-data[i]['box_height']) for i in range(len(data)) if data[i]['conf'] > 0.5]

    ResultVisiualize(ifp, msks, [0,0,1,0], coord)

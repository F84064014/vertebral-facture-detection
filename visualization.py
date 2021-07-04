import os
import copy
import numpy as np
from PIL import Image
from matplotlib import colors
from matplotlib import pyplot as plt
from torch._C import dtype

from yoloDetect import read_label


cmap = colors.ListedColormap(['blue', 'red', 'green'])
bounds=[-1.1,0.1,1.1,3]
norm = colors.BoundaryNorm(bounds, cmap.N)

'''
img: input full image
masks: list of prediction result for segmentation
conds: list of integer represent the cnodition of vertex
'''
def ResultVisiualize(img_fp, msks, vconds, sconds, boxs, lst_plot_points=None, dest_fp=None):
    
    print(vconds)

    plt.clf()
    img = Image.open(img_fp)
    plt.figure(figsize=(10,10*img.size[1]/img.size[0]))
    plt.imshow(img, cmap='gray')
    plt.gca().set_axis_off()
    
    for msk, box, vcond, scond in zip(msks, boxs, vconds, sconds):

        # new image is outimg with padding = 0
        new_img = np.zeros(np.asarray(img).shape)

        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                new_img[int(box[1]+i),int(box[0]+j)] = msk[i,j]

        #　color map
        # 0: blue
        # 1: red
        # 2: green
        col = 0 if vcond==1 else 1
        col = col if scond==0 else 2
        masked = np.where(new_img > 0.8, col, np.nan)
        # masked = np.stack((masked,)*3, axis=-1)

        plt.imshow(masked, alpha=0.5, cmap=cmap, norm=norm)
        # plt.text(box[0], box[1], 'this is good')

        # cs = minCornerDistance(msk, 0)

    if lst_plot_points != None:

        lst_corners, lst_uppers, lst_lowers = lst_plot_points

        lst_corners = corner_correct(lst_corners, boxs)
        lst_uppers = corner_correct(lst_uppers, boxs)
        lst_lowers = corner_correct(lst_lowers, boxs)

        for corners, uppers, lowers in zip(lst_corners, lst_uppers, lst_lowers):
            for corner in corners:
                plt.plot(corner[0], corner[1], 'r+')
            idx_mid = len(uppers)//2
            plt.plot(uppers[idx_mid][0], uppers[idx_mid][1], 'g+')
            plt.plot(lowers[idx_mid][0], lowers[idx_mid][1], 'g+')
            plt.plot([uppers[idx_mid][0], lowers[idx_mid][0]], [uppers[idx_mid][1], lowers[idx_mid][1]], 'y-')

        for corners, uppers, lowers in zip(lst_corners, lst_uppers, lst_lowers):
            plt.plot([corners[0][0], corners[1][0]], [corners[0][1], corners[1][1]], 'y-')
            plt.plot([corners[3][0], corners[2][0]], [corners[3][1], corners[2][1]], 'y-')



    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if dest_fp != None:
        plt.savefig(dest_fp, bbox_inches = 'tight', pad_inches = 0)
    else:
       plt.show()


'''
lst_corners: list of corners=[(x1,y1), ..., (x4,y4)]
boxs: list of boxx=(left, top, right, bot)

return: list of new corner with corrected coordinate
'''
def corner_correct(lst_corners, boxs):

    new_lst_corners = []
    for corners, box in zip(lst_corners, boxs):

        x0, y0 = box[0], box[1]
        new_lst_corners.append([(x0+corners[i][0], y0+corners[i][1]) for i in range(len(corners))])

    return new_lst_corners



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

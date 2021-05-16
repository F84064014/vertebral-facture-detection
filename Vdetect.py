import os
import argparse
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from visualization import ResultVisiualize
from UnetDetect import getUnet, shape_detect
from yoloDetect import seg_by_label, executeYolo
from fractDetect import fracturePredict, get_fmodel
from UnetDetect import UNet, Up, DoubleConv, OutConv, Down

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='description1')

    parser.add_argument('-s', '--source', type=str)
    parser.add_argument('-d', '--dest', type=str)
    parser.add_argument('-n', '--name', type=str, default='exp')

    args = parser.parse_args()
    print(args.source)
    print(args.dest)
    print(args.name)

    if os.path.isfile(args.source):
        print('source file: {}'.format(args.source))
    elif os.path.isdir(args.source):
        print('source folder: {}'.format(args.source))
    else:
        raise FileNotFoundError('file {} not found'.format(args.source))

    if os.path.isdir(args.dest):
        print('dest file: {}'.format(args.source))
    else:
        raise FileNotFoundError('file {} not found'.format(args.dest))

    executeYolo(src=args.source, dest=args.dest, name=args.name)

    # fps = if input single image path [fp1], else if folder path [fp1, fp2, pf3...]
    # src_fps: list of filepath of images
    # obj_fps: list of yolo format file
    src_fps = [args.source] if not os.path.isdir(args.source) else [os.path.join(args.source, fn) for fn in os.listdir(args.source) if fn.endswith('.bmp')]
    obj_fps = [os.path.join(args.dest, args.name, 'labels', fn) for fn in os.listdir(os.path.join(args.dest, args.name, 'labels'))]

    src_fps.sort(key=lambda s: s.split('/')[-1])
    obj_fps.sort(key=lambda s: s.split('/')[-1])

    # print(src_fps)
    # print(obj_fps)

    # =========== need more flexable model====================
    vertebral_conditions = []

    md1 = getUnet(md_path = r'C:\Users\user\OneDrive\桌面\vertex\weights\Unet\unet0424H.opt')
    md2 = getUnet(md_path = r'C:\Users\user\OneDrive\桌面\vertex\weights\Unet\unet0425H.opt')
    fmds = get_fmodel(path = r'C:\Users\user\OneDrive\桌面\vertex\MLmodel.pickle')
    # =========================================================

    # path of segmented mask
    os.mkdir(os.path.join(args.dest, args.name, 'masks'))
    # path of features and detection result
    os.mkdir(os.path.join(args.dest, args.name, 'features'))
    # path of output image
    os.mkdir(os.path.join(args.dest, args.name, 'visuals'))

    report_path = os.path.join(args.dest, args.name, 'report.txt')
    mask_root = os.path.join(args.dest, args.name, 'masks')
    feature_root = os.path.join(args.dest, args.name, 'features')
    visual_root = os.path.join(args.dest, args.name, 'visuals')

    # foreach full_Xray
    for img_fp, obj_fp in zip(src_fps, obj_fps):

        preds = []
        pred1 = []
        pred2 = []

        # read the result of yolo
        seg_imgs, boxs, positions = seg_by_label(img_fp, obj_fp)

        # foreach vertext in a full_Xray
        for idx, seg_img in enumerate(seg_imgs):
            pred1.append(shape_detect(seg_img, md1, Snake=False, save_path=None, img_size=224, t=0.7))

        for idx, seg_img in enumerate(seg_imgs):
            pred2.append(shape_detect(seg_img, md2, Snake=False, save_path=None, img_size=256, t=0.7))

        preds = [(p1 + p2)/2 for p1, p2 in zip(pred1, pred2)]

        for i, (pred, position) in enumerate(zip(preds, positions)):

            case_id = obj_fp.split('\\')[-1].split('.')[0]
            feature_path = os.path.join(feature_root, case_id + '.txt')
            mask_path = os.path.join(mask_root, case_id + '_' + position + '.txt')

            np.savetxt(mask_path, pred, fmt='%d')

            if i != 0 and i != len(preds):
                pass

            # with open(os.path.join(args.dest, args.name, 'preds', 'test{}.txt'.format(i)))    
            vertebral_conditions.append(fracturePredict(mask = pred, position = position, mds=fmds, outfp=feature_path))

        visual_path = os.path.join(visual_root, case_id + '.jpg')
        ResultVisiualize(img_fp, preds, vertebral_conditions, boxs, visual_path)

        print('{} finished. total {} vertexs'.format(img_fp.split('\\')[-1], len(positions)))
        print('report:')

        with open(report_path, 'a') as report:

            report.write('src: {}\n'.format(img_fp))

            for position, vertebral_condition in zip(positions, vertebral_conditions):
                vertebral_cond = 'normal' if vertebral_condition==1 else 'fracture'
                print('{}: {}'.format(position, vertebral_cond))
                report.write('{}: {}\n'.format(position, vertebral_cond))
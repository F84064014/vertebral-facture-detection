import os
import shutil
import argparse
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from torch._C import dtype

from segfunct import get_angle
from visualization import ResultVisiualize
from UnetDetect import getUnet, shape_detect
from yoloDetect import seg_by_label, executeYolo
from fractDetect import fracturePredict, get_fmodel
from screwDetect import screwPredict, get_scew_model

from screwDetect import BuildModel
from UnetDetect import UNet, Up, DoubleConv, OutConv, Down

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='description1')

    parser.add_argument('-s', '--source', type=str)
    parser.add_argument('-d', '--dest', type=str)
    parser.add_argument('-n', '--name', type=str, default='exp')
    parser.add_argument('-ma', '--mask', type=str, default=None)
    parser.add_argument('-me', '--method', type=str, default='rfc')
    parser.add_argument('-o', '--object', type=str, default=None)

    args = parser.parse_args()
    print(args.source)
    print(args.dest)
    print(args.name)
    print(args.mask)

    # check source file or folder exist
    if os.path.isfile(args.source):
        print('source file: {}'.format(args.source))
    elif os.path.isdir(args.source):
        print('source folder: {}'.format(args.source))
    else:
        raise FileNotFoundError('file {} not found'.format(args.source))

    # check destination exist
    if os.path.isdir(args.dest):
        print('dest file: {}'.format(args.source))
    else:
        raise FileNotFoundError('file {} not found'.format(args.dest))

    # check if predicted mask at prior offered
    if args.mask != None:
        if os.path.isdir(args.mask):
            print('Mask segmentation result from {}'.format(args.mask))
        else:
            raise FileNotFoundError('file {} not found'.format(args.mask))

    if args.object != None:
        if os.path.isdir(args.object):
            print('Object detection result file from {}')
            os.makedirs(os.path.join(args.dest, args.name, 'labels'))
            for obj_fn in os.listdir(args.object):
                shutil.copyfile(os.path.join(args.object, obj_fn), os.path.join(args.dest, args.name, 'labels', obj_fn))
        else:
            raise FileNotFoundError('file {} not found'.format(args.object))
    else:
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

    md1 = getUnet(md_path = r'C:\Users\user\OneDrive\桌面\vertex\weights\Unet\unet0424H.opt')
    md2 = getUnet(md_path = r'C:\Users\user\OneDrive\桌面\vertex\weights\Unet\unet0425H.opt')
    fmds = get_fmodel(path = r'C:\Users\user\OneDrive\桌面\vertex\MLmodel.pickle')
    screwmd = get_scew_model(path=r'C:\Users\user\OneDrive\桌面\vertex\weights\Alexnet\screw_clf(2).pt')
    # =========================================================

    # path of segmented mask
    os.mkdir(os.path.join(args.dest, args.name, 'masks'))
    # path of features and detection result
    os.mkdir(os.path.join(args.dest, args.name, 'features'))
    # path of output image
    os.mkdir(os.path.join(args.dest, args.name, 'visuals'))
    
    # read from history result
    pre_mask_root = args.mask
    pre_objd_root = args.object

    report_path = os.path.join(args.dest, args.name, 'report.txt')
    mask_root = os.path.join(args.dest, args.name, 'masks')
    feature_root = os.path.join(args.dest, args.name, 'features')
    visual_root = os.path.join(args.dest, args.name, 'visuals')
    method = args.method

    # foreach full_Xray
    for img_fp, obj_fp in zip(src_fps, obj_fps):
        
        # ========required variable========
        preds = []
        pred1 = []
        pred2 = []

        vertebral_conditions = []
        selected_corners = []
        selected_uppers = []
        selected_lowers = []
        # =================================

        # read the result of yolo
        seg_imgs, boxs, positions = seg_by_label(img_fp, obj_fp)
        # get the rotation degree
        degrees = get_angle(boxs)

        # foreach vertext in a full_Xray
        # when there is no predicted mask in prior project
        # I found that use single model might works better
        if args.mask == None:
            # for idx, seg_img in enumerate(seg_imgs):
            #     pred1.append(shape_detect(seg_img, md1, Snake=False, save_path=None, img_size=224, t=0.7))

            for idx, seg_img in enumerate(seg_imgs):
                pred2.append(shape_detect(seg_img, md2, Snake=False, save_path=None, img_size=256, t=0.7))

            # preds = [(p1 + p2)/2 for p1, p2 in zip(pred1, pred2)]
            preds = [p2 for p2 in pred2]

        # predicted mask in prior porject
        else:
            for position in positions:
                case_id = obj_fp.split('\\')[-1].split('.')[0]
                feature_path = os.path.join(feature_root, case_id + '.txt')
                mask_path = os.path.join(pre_mask_root, case_id + '_' + position + '.txt')
                preds.append(np.loadtxt(mask_path, dtype=np.uint8))

        for i, (pred, position, degree) in enumerate(zip(preds, positions, degrees)):

            case_id = obj_fp.split('\\')[-1].split('.')[0]
            feature_path = os.path.join(feature_root, case_id + '.txt')
            mask_path = os.path.join(mask_root, case_id + '_' + position + '.txt')

            np.savetxt(mask_path, pred, fmt='%d')

            # with open(os.path.join(args.dest, args.name, 'preds', 'test{}.txt'.format(i)))
            # corners: [(xn,yn) for n in 4]
            # condition: 1 or 0
            corners, uppers, lowers, cond = fracturePredict(
                mask = pred, 
                position = position, 
                mds=fmds, 
                angle=-degree, 
                method=method, 
                outfp=feature_path
                )
            vertebral_conditions.append(cond)
            selected_corners.append(corners)
            selected_uppers.append(uppers)
            selected_lowers.append(lowers)

        screw_conditions = screwPredict(seg_imgs, screwmd)

        visual_path = os.path.join(visual_root, case_id + '.jpg')
        selected_sets = [selected_corners, selected_uppers, selected_lowers]
        ResultVisiualize(img_fp, preds, vertebral_conditions, screw_conditions, boxs, selected_sets, visual_path)


        print('{} finished. total {} vertexs'.format(img_fp.split('\\')[-1], len(positions)))
        print('report:')

        with open(report_path, 'a') as report:

            report.write('src: {}\n'.format(img_fp))

            for position, vertebral_condition in zip(positions, vertebral_conditions):
                vertebral_cond = 'normal' if vertebral_condition==1 else 'fracture'
                print('{}: {}'.format(position, vertebral_cond))
                report.write('{}: {}\n'.format(position, vertebral_cond))
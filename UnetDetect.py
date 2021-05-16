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

# model config

def getUnet(md_path = r'C:\Users\user\OneDrive\桌面\vertex\weights\Unet\unet0328H.opt'):

    model = torch.load(md_path, map_location=torch.device('cpu'))['model']
    return model

'''
seg_img: cropped image of single vetebral
model: the detection model
Histeq: if True do histogram equalization trainsform
Snake: if True do active contour model
save_path: if None dont save, else save to save_path
t: threshold for Unet
'''

def shape_detect(seg_img, model, Histeq=True, Snake=True, save_path=None, img_size=224, t=0.8):

    assert model != None, 'model doesn\'t exist'

    # Histogram equalization
    if Histeq == True:
        input_img = Image.fromarray(equalizeHist(np.array(seg_img)))
    else:
        input_img = seg_img

    # input transformation
    input_img = TF.resize(input_img, [img_size,img_size])
    input_img = TF.to_tensor(input_img)

    # detection (no gpu so just do one by one)
    model.eval()
    output = model(input_img.unsqueeze(0))
    prob = torch.sigmoid(output)

    prob = TF.resize(prob, [*(seg_img.size)][::-1])
    outimg = prob.squeeze().detach().numpy()

    # masked = np.ma.masked_where(outimg < 0.8 , outimg)
    masked = np.where(outimg > t, 1, 0)

    # tunning with active contour model
    if Snake:
        s = np.linspace(0, 2*np.pi, 400)
        r = outimg.shape[0]/2 + 350*np.sin(s)
        c = outimg.shape[1]/2 + 350*np.cos(s)
        init = np.array([r, c]).T

        # beta = the smoothness
        snake = active_contour(gaussian(outimg,sigma=3),
                            init, alpha=0.075, beta=5, gamma=0.001)

        outline = Image.new(size=outimg.shape[::-1], mode='1')
        ImageDraw.Draw(outline).polygon(
            [(int(s[1]), int(s[0])) for s in snake],
            outline=1,
            fill=1
        )
        # masked = np.ma.masked_where(np.array(outline) < 0.8, np.array(outline))
        masked = np.where(outimg > t, 1, 0)

    if save_path != None:
        np.savetxt(save_path, masked, fmt='%d')
    # elif save_path != None:
        # raise Warning('fail to save the mask to {}'.format(save_path))

    return masked


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = TF.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

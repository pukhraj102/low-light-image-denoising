import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *

def load_image(filename):
    data_lowlight = Image.open(filename)
    resize_transform = transforms.Resize((256, 256))  # Define the resize transform
    data_lowlight = resize_transform(data_lowlight)  # Apply the resize transform
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)
    return data_lowlight

def psnr_calc(target, output, max_pixel=1.0):
    mse = torch.mean((target - output) ** 2)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# def psnr_img(img1, img2):
#     img1 = img1.convert('RGB')
#     img2 = img2.convert('RGB')
#
#     img1_array = np.array(img1)
#     img2_array = np.array(img2)
#
#     if img1_array.shape != img2_array.shape:
#         raise ValueError("Input images must have the same dimensions.")
#
#     # Calculate Mean Squared Error (MSE)
#     mse = np.mean((img1_array - img2_array) ** 2)
#     if mse == 0:
#         return float('inf')
#
#     # Calculate PSNR
#     max_pixel = 255.0
#     PSNR = 20 * np.log10(max_pixel / np.sqrt(mse))
#     return PSNR

def mae_img(img1, img2):
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    if img1_array.shape != img2_array.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Calculate Mean Squared Error (MSE)
    mae = np.mean(np.absolute(img1_array - img2_array))
    np.sum(np.abs(img1_array - img2_array)) / np.sum(img1_array + img2_array)
    return mae

def mse_img(img1, img2):
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    if img1_array.shape != img2_array.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1_array - img2_array) ** 2)
    if mse == 0:
        return float('inf')
    else:
      return mse



def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()




#utilities for model.py

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)






import torch
import torch.nn as nn
import torchvision
import os
import model
import numpy as np
from PIL import Image
import glob
from model import *
from utils import *

test_low = sorted(glob.glob("./test/low/*"))
test_high = sorted(glob.glob("./test/high/*"))

def infer(image_path, color_net, predicted_path, save_enable):

    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0)

    with torch.no_grad():
        gray, color_hist, enhanced_image= color_net(data_lowlight)

    if save_enable:
        filename = os.path.basename(image_path)
        save_path = predicted_path + filename  # Construct the full save path
        torchvision.utils.save_image(enhanced_image, save_path)

    return enhanced_image

def perform_test(predicted_path, save_enable):
    pretrain_path = './weights.pth'

    with torch.no_grad():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        color_net = model.color_net()
        color_net = nn.DataParallel(color_net)
        color_net = color_net
        color_net.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cpu')))

        for image in test_low:
            print(image)
            infer(image, color_net, predicted_path, save_enable)


def check_psnr(predicted_path, plot):

    psnr_values = []
    mse_values = []
    mae_values = []

    test_low_light_images = sorted(glob.glob("./test/low/*"))
    test_enhanced_images = sorted(glob.glob("./test/high/*"))
    result_image = sorted(glob.glob(predicted_path+"*"))

    for idx in range(len(test_low)):
        original_image = Image.open(test_low[idx])
        enhanced_image = Image.open(result_image[idx])
        ground_truth = Image.open(test_high[idx])

        if plot:
            plot_results(
                [original_image, enhanced_image, ground_truth],
                ["Original", "Enhanced", "Ground Truth"],
                (12, 12),
            )

        PSNR = psnr_calc(enhanced_image, ground_truth)
        MSE = mse_img(enhanced_image, ground_truth)
        MAE = mae_img(enhanced_image, ground_truth)
        psnr_values.append(PSNR)
        mse_values.append(MSE)
        mae_values.append(MAE)
        print("PSNR is: ", PSNR, "\tMSE IS: ", MSE, "\tMAE is: ", MAE, "\n")


    average_psnr = sum(psnr_values) / len(psnr_values)
    average_mse = sum(mse_values) / len(mse_values)
    average_mae = sum(mae_values) / len(mae_values)

    print(" Average PSNR is: ", average_psnr, "\nAverage MSE IS: ", average_mse, "\nAverage MAE is: ", average_mae,
          "\n")



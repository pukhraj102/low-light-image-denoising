import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

from model import *


def load_image(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(images=image, size=[256, 256])
    image = image / 255.0
    return image


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

def psnr_img(img1, img2):
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

        img1_array = np.array(img1)
        img2_array = np.array(img2)

        if img1_array.shape != img2_array.shape:
            raise ValueError("Input images must have the same dimensions.")
        return tf.image.psnr(img1_array, img2_array, max_val=255.0)

def mae_img(img1, img2):
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    if img1_array.shape != img2_array.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Calculate Mean Squared Error (MSE)
    mae = np.mean(np.absolute(img1_array - img2_array))
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


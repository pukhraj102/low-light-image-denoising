from glob import glob

from utils import *
from model import *

train_low_data_names = glob('./train/low/*.png')
train_low_data_names.sort()

train_low = []

for idx in range(len(train_low_data_names)):
    im = load_image(train_low_data_names[idx])
    train_low.append(im)

train_high_data_names = glob('./train/high/*.png')
train_high_data_names.sort()

train_high = []

for idx in range(len(train_high_data_names)):
    im = load_image(train_high_data_names[idx])
    train_high.append(im)

val_low = train_low[-85:]
train_low = train_low[:-85]

val_high = train_high[-85:]
train_high = train_high[:-85]

def augment_image(low_img, high_img):
    # Randomly adjust brightness to simulate varying exposure
    low_img = tf.image.random_brightness(low_img, max_delta=0.12)
    high_img = tf.image.random_brightness(high_img, max_delta=0.12)
    return low_img, high_img

# Create the training dataset with augmentation
def load_and_augment(low_img, high_img):
    low_img, high_img = augment_image(low_img, high_img)
    return low_img, high_img

dataset = tf.data.Dataset.from_tensor_slices((train_low, train_high))
dataset = dataset.map(load_and_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(16, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((val_low, val_high))
val_dataset = val_dataset.batch(16, drop_remainder=True)



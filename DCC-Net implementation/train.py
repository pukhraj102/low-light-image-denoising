import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.optim as optim
from model import color_net
from utils import *

# Define data directories
train_low_dir = './train/low'
train_high_dir = './train/high'
val_low_dir = './val/low'
val_high_dir = './val/high'



# Load training data
train_low_data_names = glob.glob(train_low_dir + '/*.png')
train_low_data_names.sort()

train_low = []
for idx in range(len(train_low_data_names)):
    im = load_image(train_low_data_names[idx])
    train_low.append(im)

train_high_data_names = glob.glob(train_high_dir + '/*.png')
train_high_data_names.sort()

train_high = []
for idx in range(len(train_high_data_names)):
    im = load_image(train_high_data_names[idx])
    train_high.append(im)

# Split data into training and validation sets
val_low = train_low[-85:]
train_low = train_low[:-85]

val_high = train_high[-85:]
train_high = train_high[:-85]

train_low_tensor = torch.stack(train_low)
train_high_tensor = torch.stack(train_high)

val_low_tensor = torch.stack(val_low)
val_high_tensor = torch.stack(val_high)

# Create data loaders
batch_size = 16
val_batch_size = 85
train_dataset = torch.utils.data.TensorDataset(train_low_tensor, train_high_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_low_tensor, val_high_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)


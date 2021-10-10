import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split  # Batch data
from torchvision.utils import make_grid     # Display images in a grid format
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt

import cv2

# data_dir = './data/Cat_Dog_data'
# print(os.listdir(data_dir))

data_dir ='C:/Users/Study1/Documents/Programming/Artificial Intelligence/My AI Projects/Datasets/Dataset_BUSI_with_GT'
print(os.listdir(data_dir))
classes = os.listdir(data_dir)
print(classes)

#Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyperparameters
num_workers = 4     # Makes loading of batch easier, ensures all the cores of the CPU used
batch_size = 64
pin_memory = True   # Keeps block of memory saved for each batch
learning_rate = 0.0001
num_layers = 2
num_classes = 10
num_epochs = 4

def import_images(folder,target):
    images = []
    for item in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,item),0)
        if img is not None:
            images.append([img,target])
    return images


benign = import_images(data_dir + "/benign",0)
malignant = import_images(data_dir + "/malignant/",1)
normal = import_images(data_dir + "/normal/",2)
benign.extend(malignant)
benign.extend(normal)
plt.imshow(benign[1][0])
plt.show()





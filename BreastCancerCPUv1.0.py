import numpy as np
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
# print(os.listdir(data_dir))
classes = os.listdir(data_dir)
# print(classes)

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

#Function to return an array of grey scale images
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

# Load all images into one array
benign.extend(malignant)
benign.extend(normal)
# plt.imshow(benign[0][0])
# plt.show()

full_data = benign
feature_matrix = []
label = []
for x, y in full_data:
    feature_matrix.append(x)
    label.append(y)

print(feature_matrix[0])
plt.imshow(feature_matrix[0])
# plt.show()
# print(label[0])

X = []
img_size = 128

for x in feature_matrix:
    new_array = cv2.resize(x,(img_size, img_size))
    X.append(new_array)

# plt.imshow(X[0])
# plt.show()

X_corrected = []
for image in X:
    image = image/255
    X_corrected.append(image)
# ********************************************************
# plt.imshow(X_corrected[0])
# plt.show()
print(np.array(X_corrected).shape)
X_M = np.array(X_corrected)
print(X_M.shape[0])

print(np.array(X_corrected).shape)
# print(np.array(X_corrected).shape[0])
# print(np.array(X_corrected).shape[1])
# print(np.array(X_corrected).shape[2])
#
# X_M_R = X_M.reshape(X_M.shape[0], X_M.shape[1],X_M.shape[2],1)
# print(X_M_R)



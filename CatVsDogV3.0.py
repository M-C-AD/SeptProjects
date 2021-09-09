import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_dir = './data/Cat_Dog_data'
print(data_dir)

#Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyperparameters
#num_workers = 4
batch_size = 64
# pin_memory = True
# load_model = True
# save_model = True
# weight_decay = 0.0001
learning_rate = 0.0001
num_layers = 2
num_classes = 10
num_epochs = 2










import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms

data_dir = './data/Cat_Dog_data'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + '/train')
print(classes)

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

# Data transformers
train_transform = transforms([transforms.Resize(28),
                              transforms.ToTensor(),
                              transforms.Normalize([0.5, 0.5, 0.5],
                                                   [0.5, 0.5, 0.5])])
test_transform = transforms([transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5])])

#Data Transformers
train_data = datasets.ImageFolder(data_dir + 'train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + 'test', transform=test_transform)

#Load data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

print(train_loader)
print(test_loader)









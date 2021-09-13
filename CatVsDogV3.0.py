import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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
train_transform = transforms.Compose([transforms.Resize((28, 28)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

#Load data
train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size= batch_size)

print(train_loader)
print(test_loader)

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conV1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conV2 = nn.Conv2d(32, 16, 3)
        self.fc1 = nn.Linear(16*11*11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conV1(x)))
        x








import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader     # Batch data
from torchvision.utils import make_grid     # Display images in a grid format
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt

data_dir = './data/Cat_Dog_data'
# print(os.listdir(data_dir))

classes = os.listdir(data_dir + '/train')
# print(classes)

#Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyperparameters
num_workers = 4     # Makes loading of batch easier, ensures all the cores of the CPU used
batch_size = 64
pin_memory = True   # Keeps block of memory saved for each batch
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

# Load data
train_Dloader = DataLoader(train_data, batch_size= batch_size, shuffle=True)
test_Dloader = DataLoader(test_data, batch_size= batch_size)
print(train_Dloader)
print(test_Dloader)

# Display a grid of the images
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


show_batch(train_Dloader)
plt.show()

# img, label = train_data[0]
# print(img.shape, label)
# print(train_data.classes)
# print(img)

# def show_example(img, label):
#     print('Label:', train_data.classes[label], "(" + str(label) + ")")
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()
#
#
# img, label = train_data[0]
# show_example(img, label)

# Base image classification model
class BaseImageClassificationModel(nn.Module):
    def train_step(self, batch):
        images, labels = batch
        out = self(images)                      # Generate predictions
        loss = F.cross_entropy(out, labels)     # Calculate loss


class CDNet(nn.Module):
    def __init__(self):
        super(CDNet, self).__init__()
        self.conV1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conV2 = nn.Conv2d(32, 16, 3)
        self.fc1 = nn.Linear(16*11*11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conV1(x)))
        x = self.pool(F.relu(self.conV2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



model = CDNet().to(device)
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)







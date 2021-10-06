# install kaggle
!pip
install - q
kaggle

from google.colab import files

files.upload()

# create a kaggle folder
! mkdir
~ /.kaggle

# copy the kaggle.json to folder created
! cp
kaggle.json
~ /.kaggle /

# Permission for the json to act
! chmod
600
~ /.kaggle / kaggle.json

# to list all the datasets in kaggle
! kaggle
datasets
list

!kaggle
competitions
download - c
dogs - vs - cats

# Upload data and extract the contents
from zipfile import ZipFile

file_name = "/content/train.zip"

with ZipFile(file_name, 'r') as zip:
    zip.extractall()
    print('done')

import os

data_dir_list = os.listdir('/content/train')
# print(data_dir_list)

path, dirs, files = next(os.walk("/content/train"))
file_count = len(files)
print(file_count)

original_dataset_dir = '/content/train'
base_dir = '/content/cats_and_dogs_small'
os.mkdir(base_dir)  # make base directory

# Create directory paths

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

import shutil


def createFNameCat(org_data_dir, train_data_class_dir, rangeInput1, rangeInput2):
    "This funciton is to create the source and desitnation paths and copy the data."
    fnames = ['cat.{}.jpg'.format(i) for i in range(rangeInput1, rangeInput2)]
    for fname in fnames:
        src = os.path.join(org_data_dir, fname)
        dst = os.path.join(train_data_class_dir, fname)
        # print(src,dst)
        shutil.copyfile(src, dst)


def createFNameDog(org_data_dir, train_data_class_dir, rangeInput1, rangeInput2):
    "This funciton is to create the source and desitnation paths and copy the data."
    fnames = ['dog.{}.jpg'.format(i) for i in range(rangeInput1, rangeInput2)]
    for fname in fnames:
        src = os.path.join(org_data_dir, fname)
        dst = os.path.join(train_data_class_dir, fname)
        # print(src,dst)
        shutil.copyfile(src, dst)


createFNameCat(original_dataset_dir, train_cats_dir, 0, 10500)
createFNameCat(original_dataset_dir, validation_cats_dir, 10500, 11500)
createFNameCat(original_dataset_dir, test_cats_dir, 11500, 12500)

createFNameDog(original_dataset_dir, train_dogs_dir, 0, 10500)
createFNameDog(original_dataset_dir, validation_dogs_dir, 10500, 11500)
createFNameDog(original_dataset_dir, test_dogs_dir, 11500, 12500)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))

print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split  # Batch data
from torchvision.utils import make_grid  # Display images in a grid format
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt

data_dir = '/content/cats_and_dogs_small'
print(os.listdir(data_dir))
print('test33')

classes = os.listdir(data_dir + '/train')
print(classes)

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyperparameters
num_workers = 4  # Makes loading of batch easier, ensures all the cores of the CPU used
batch_size = 64
pin_memory = True  # Keeps block of memory saved for each batch
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
validation_transform = transforms.Compose([transforms.Resize((28, 28)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)  # original line
# entire_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
validation_data = datasets.ImageFolder(data_dir + '/validation', transform=validation_transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)
# random_seed = 42
# torch.manual_seed(random_seed)
# val_size = 5000
# train_size = len(entire_dataset) - val_size
# train_ds, val_ds = random_split(entire_dataset, [train_size, val_size])
# print('Training ds', len(train_ds), '  Validation ds', len(val_ds)) # ********************************

# Load data
train_Dloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_Dloader = DataLoader(validation_data, batch_size * 2, num_workers=4, pin_memory=True)
test_Dloader = DataLoader(test_data, batch_size=batch_size)
print(train_Dloader)
print(validation_Dloader)
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
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch[{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
              .format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class CDNet(BaseImageClassificationModel):
    def __init__(self):
        super(CDNet, self).__init__()
        self.conV1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conV2 = nn.Conv2d(32, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conV1(x)))
        x = self.pool(F.relu(self.conV2(x)))
        # print('pre flattern shape',  x.shape) #************************************************
        x = x.view(x.shape[0], -1)
        # x = x.view(-1, 16*5*5)
        # print('X shape', x.shape) #************************************************
        x = F.relu(self.fc1(x))
        # print('X shape', x.shape) #************************************************
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# #Set Device 2
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_Dloader = DeviceDataLoader(train_Dloader, device)
model = CDNet().to(device)

# Quick check of the model
for images, labels in train_Dloader:
    print('images.shape', images.shape)
    print('label', labels[0])
    # print('test 1') #****************************************
    out = model(images)
    print('out.shape', out.shape)
    print('out[0]', out[0])
    break

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)  # *************************************


# Device Management
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# print(device) # *************************
train_Dloader = DeviceDataLoader(train_Dloader, device)
validation_Dloader = DeviceDataLoader(validation_Dloader, device)
to_device(model, device)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.train_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


model = to_device(CDNet(), device)
evaluate(model, validation_Dloader)
print(model)

num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001

history = fit(num_epochs, lr, model, train_Dloader, validation_Dloader, opt_func=opt_func)


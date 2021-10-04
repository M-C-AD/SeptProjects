#install kaggle
!pip install -q kaggle

from google.colab import files
files.upload()

#create a kaggle folder
! mkdir ~/.kaggle

#copy the kaggle.json to folder created
! cp kaggle.json ~/.kaggle/

# Permission for the json to act
! chmod 600 ~/.kaggle/kaggle.json

# to list all the datasets in kaggle
! kaggle datasets list

!kaggle competitions download -c dogs-vs-cats

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
os.mkdir(base_dir) #make base directory

#Create directory paths

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
print('Validation Loader 2nd check')
show_batch(validation_Dloader)
plt.show()

# Base image classification model
class BaseImageClassificationModel(nn.Module):
    def train_step(self, batch):
        images, labels = batch
        out = self(images)                      # Generate predictions
        loss = F.cross_entropy(out, labels)     # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                      # Generate predictions
        loss = F.cross_entropy(out, labels)     # Calculate loss
        acc = accuracy(out, labels)             # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch[{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
              .format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(BaseImageClassificationModel):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.Dropout(0.5),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out1 = self.conv1(xb)
        out2 = self.conv2(out1)
        out3 = self.res1(out2) + out2
        out4 = self.conv3(out3)
        out5 = self.conv4(out4)
        out6 = self.res2(out5) + out5
        out7 = self.classifier(out6)
        return out7

# Device Management
def get_default_device():
    """Pick GPU if available else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('CPU')


def to_device(data, device):
    """Move tensors to the device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move to device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yeld a batch of data after moving to the device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# print(device) # *************************
train_Dloader = DeviceDataLoader(train_Dloader, device)
validation_Dloader = DeviceDataLoader(validation_Dloader, device)
# to_device(model, device)

# print model
model = to_device(ResNet9(3, 2), device)


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

# print model
model = to_device(ResNet9(3, 2), device)
# print('New Model\n', model)

# model = to_device(CDNet(), device)
# evaluate(model, validation_Dloader)
# initial_result = evaluate(model, validation_Dloader)
print(model)

num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001

history = fit(num_epochs, lr, model, train_Dloader, validation_Dloader, opt_func= opt_func)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    plt.show()


plot_accuracies(history)

def plot_losses(history):

    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.show()


plot_losses(history)

history1 = fit(num_epochs, lr, model,train_Dloader, validation_Dloader, opt_func= opt_func)
history2 = fit(5, 0.001, model,train_Dloader, validation_Dloader, opt_func= opt_func)
history3 = fit(5, 0.001, model,train_Dloader, validation_Dloader, opt_func= opt_func)
history4 = fit(5, 0.001, model,train_Dloader, validation_Dloader, opt_func= opt_func)

plot_accuracies(history1)
plot_losses(history1)

history = [initial_result] + history1 + history2 + history3 + history4
accuracies = [initial_result['val_acc'] for intial_result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')

img, label = test_data[0]
plt.imshow(img[0], cmap='gray')
print('Shape', img.shape)
print('Label', label)

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_data[10]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
plt.show()

img, label = test_data[100]
plt.imshow(img[0], cmap='gray')
print('Label', label, ', Predicted:', predict_image(img, model))
plt.show()

img, label = test_data[957]
plt.imshow(img[0], cmap='gray')
print('Label', label, ', Predicted:', predict_image(img, model))
plt.show()

# Save the model to use another time
torch.save(model.state_dict(), 'catVdog_CNN_28-09-21.pth')
print(model.state_dict())

# To instantiate a new model with these weights
model2 = ResNet9()
model2.load_state_dict(torch.load('catVdog_CNN_28-09-21.pth'))
print(model2.state_dict())
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
                                      # transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
                                      # transforms.RandomHorizontalFlip(),
                                      # transforms.RandomRotation(20),
                                      #transforms.RandomResizedCrop(256, scale=(0.5, 0.9), ratio=(1, 1)),
                                      # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])


test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform) # original line
entire_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transform) # original line
test_data = datasets.ImageFolder(data_dir + '/test',
                                 train=False,
                                 transform=test_transform)
random_seed = 42
torch.manual_seed(random_seed)
val_size = 1000
train_size = len(entire_dataset) - val_size
train_ds, val_ds = random_split(entire_dataset, [train_size, val_size])
# print('Training ds', len(train_ds), '  Validation ds', len(val_ds)) # ********************************

# Load data
train_Dloader = DataLoader(train_ds, batch_size= batch_size, shuffle=True)
validation_Dloader = DataLoader(val_ds, batch_size * 2) #, num_workers=4, pin_memory=True)
test_Dloader = DataLoader(test_data, batch_size= batch_size)
print('Training Loader\n')
print(train_Dloader)
print('Validation Loader\n')
print(validation_Dloader)
print('Test Loader\n')
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


class CDNet(BaseImageClassificationModel):
    def __init__(self):
        super(CDNet, self).__init__()
        self.conV1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conV2 = nn.Conv2d(32, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        out = self.pool(F.relu(self.conV1(x)))
        out = self.pool(F.relu(self.conV2(out)))
        # print('pre flattern shape',  out.shape) #************************************************
        out = out.view(out.shape[0], -1)
        # out = out.view(-1, 16*5*5)
        # print('out shape', out.shape) #************************************************
        out = F.relu(self.fc1(out))
        # print('X shape', x.shape) #************************************************
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out  # Adding x turns this into a residual block, to improve model performance


# model = CDNet().to(device)

# # Quick check of the model
# for images, labels in train_Dloader:
#     print('images.shape', images.shape)
#     print('label', labels[0])
#     # print('test 1') #****************************************
#     out = model(images)
#     print('out.shape', out.shape)
#     print('out[0]', out[0])
#     break
#
# criterian = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# print(model) #*************************************


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


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    plt.show()


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


model = to_device(CDNet(), device)
# evaluate(model, validation_Dloader)
initial_result = evaluate(model, validation_Dloader)
# print(model)

num_epochs = 5
opt_func = torch.optim.Adam
lr = 0.001

history1 = fit(num_epochs, lr, model,train_Dloader, validation_Dloader, opt_func= opt_func)
plot_accuracies(history1)
plot_losses(history1)
history2 = fit(5, 0.001, model,train_Dloader, validation_Dloader, opt_func= opt_func)
history3 = fit(5, 0.001, model,train_Dloader, validation_Dloader, opt_func= opt_func)
history4 = fit(5, 0.001, model,train_Dloader, validation_Dloader, opt_func= opt_func)

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
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


img, label = test_data[10]
plt.imshow(img[0], cmap='gray')
print('Label', label, ', Predicted:', predict_image(img, label))
plt.show()

img, label = test_data[100]
plt.imshow(img[0], cmap='gray')
print('Label', label, ', Predicted:', predict_image(img, label))
plt.show()

img, label = test_data[957]
plt.imshow(img[0], cmap='gray')
print('Label', label, ', Predicted:', predict_image(img, label))
plt.show()

test_results = evaluate(model, test_Dloader)
print(test_results)

test_loader = DeviceDataLoader(DataLoader(test_data, batch_size*2), device)
result = evaluate(model, test_loader)
print('Batch test results\n', result)

# Save the model to use another time
torch.save(model.state_dict(), 'catVdog_CNN_28-09-21.pth')
print(model.state_dict())

# To instantiate a new model with these weights
model2 = CDNet()
model2.load_state_dict(torch.load('catVdog_CNN_28-09-21.pth'))
print(model2.state_dict())




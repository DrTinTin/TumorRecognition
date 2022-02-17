import datetime
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_path = '../dataSource/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))]))

cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))]))

# index the two class_names (0 and 2)
label_map = {0: 0, 2: 1}
training_set = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
validation_set = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_channels)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, img):
        out = self.conv(img)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + img


class NetResDeep(nn.Module):
    def __init__(self, n_chans=32, n_blocks=10):
        super(NetResDeep, self).__init__()
        self.n_chans = n_chans
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*(n_blocks * [ResBlock(n_channels=n_chans)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, img):
        out = fun.max_pool2d(torch.relu(self.conv1(img)), 2)
        out = self.resblocks(out)
        out = fun.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans)
        out = torch.relu((self.fc1(out)))
        out = self.fc2(out)
        return out


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
model = NetResDeep(n_chans=32, n_blocks=100).to(device=device)
model.train()
train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss_fun = nn.CrossEntropyLoss()


def training_loop(epochs, optimizer, model, loss_fun, train_loader):
    for epoch in range(epochs):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fun(outputs, labels)
            optimizer.zero_grad()
            loss.backward()  # compute the gradients of all parameters
            optimizer.step()  # updates the model
            loss_train += loss.item()  # transform the loss to a Python number with .item(),to escape the gradients.
        if epoch % 10 == 0:
            # get the average loss per batch
            print(f'{datetime.datetime.now()} Epoch {epoch}, Training loss {loss_train / len(train_loader)}')


training_loop(
    epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fun=loss_fun,
    train_loader=train_loader
)

train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)
model.eval()


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)  # predicted is the predicted index
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print(f"Accuracy {name}: {correct / total:.2f}")  # f-string采用 {content:format} 设置字符串格式


validate(model, train_loader, val_loader)

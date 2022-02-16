import datetime

from torchvision import datasets, transforms
import numpy as np
import collections
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


class Net_threeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, img):
        out = fun.max_pool2d(torch.tanh(self.conv1(img)), 2)
        out = fun.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
model = Net_threeLayer().to(device=device)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fun = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)


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
torch.save(model.state_dict(), data_path + 'birds_vs_airplanes.pt') # save the model
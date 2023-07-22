"""
@Description: Implementing a deep CNN using PyTorch
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-20 22:00:50
"""

import torchvision
from torchvision import transforms
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True, transform=transform, download=False
)
import torch
from torch.utils.data import Subset
mnist_valid_dataset = Subset(mnist_dataset,
                             torch.arange(10_000))
mnist_train_dataset = Subset(
    mnist_dataset, torch.arange(10_000, len(mnist_dataset)))
mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path, train=False, transform=transform, download=False
)

from torch.utils.data import DataLoader
batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

import torch.nn as nn
model = nn.Sequential()
model.add_module('cov1', nn.Conv2d(
    in_channels=1, out_channels=32, kernel_size=5, padding=2))

model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('cov2', nn.Conv2d(
    in_channels=32, out_channels=64, kernel_size=5, padding=2))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))

# 可以更为简单的计算特征图大小
x = torch.ones((4, 1, 28, 28))
print(model(x).shape)  # torch.Size([4, 64, 7, 7])

# 因为全连接必须是二维的，即(batch_size * input_units)
model.add_module('flatten', nn.Flatten())
print(model(x).shape)  # torch.Size([4, 3136])

model.add_module('fc1', nn.Linear(3136, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=.5))
model.add_module('fc2', nn.Linear(1024, 10))

# 因为nn.CrossEntropyLoss()包含了softmax激活函数，因此不需要在输出层之后添加softmax激活函数
loss_fn = nn.CrossEntropyLoss()
# Adam算法的优势在于能够根据梯度变化的平均值更新步长
# The key advantage of Adam is in the choice of update step size derived from the running
# average of gradient moments.
optimizer = torch.optim.Adam(model.parameters(), lr=.001)


def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        # 在训练模型时添加 train()
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        # 在模型评估时添加 eval()
        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        print(
            f'Epoch {(epoch + 1):2d} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


torch.manual_seed(1)
num_epochs = 20
hist = train(model, num_epochs, train_dl, valid_dl)

import matplotlib.pyplot as plt
import numpy as np
x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121)
ax.plot(x_arr, hist[0], '-o', label='Train Loss')
ax.plot(x_arr, hist[1], '-->', label='Validation Loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(122)
ax.plot(x_arr, hist[2], '-o', label='Train Acc.')
ax.plot(x_arr, hist[3], '-->', label='Validation Acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()


pred = model(mnist_test_dataset.data.unsqueeze(1) / 225.)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()

is_wrong = (torch.argmax(pred, dim=1) !=
            mnist_test_dataset.targets).nonzero().squeeze()

print(f'Test Accuracy: {is_correct.mean():.4f}')

fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i + 1)
    ax.set_xticks([])
    # 第 i 个元素是一个元组，(data, target), the shape of data is 1*28*28
    ax.set_yticks([])
    img = mnist_test_dataset[i][0][0, :, :]
    pred = model(img.unsqueeze(0).unsqueeze(1))
    y_pred = torch.argmax(pred)
    ax.imshow(img, cmap='gray_r')
    ax.text(.9, .1, y_pred.item(), size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()

# predict wrong examples
fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    img = mnist_test_dataset[is_wrong[i]][0][0, ::]
    # 这一行很奇怪，似乎得到不到和之前 pred 预测结果一样的
    # pred = model(img.unsqueeze(0).unsqueeze(1))
    y_pred = torch.argmax(pred[is_wrong[i]])
    print(y_pred)
    ax.imshow(img, cmap='gray_r')
    ax.text(.9, .1, y_pred.item(), size=15, color='red',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    print(mnist_test_dataset[is_wrong[i]][1])
    ax.text(.1, .9, mnist_test_dataset[is_wrong[i]][1],
            size=15, color='green',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

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
    pass

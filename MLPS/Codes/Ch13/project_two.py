"""
@Description: Project two – classifying MNIST handwritten digits
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-13 17:19:55
"""

import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor()
])

# 首次需要将download设置为True
mnist_train_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=False
)
mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path, train=False, transform=transform, download=False
)

batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)

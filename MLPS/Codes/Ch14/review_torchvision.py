"""
@Description: recap how to read an image
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-19 20:02:51
"""

import torch
from torchvision.io import read_image
img = read_image('example-image.png')
print('Image shape', img.shape)

print('Number of channels', img.shape[0])

print('Image data type:', img.dtype)

print(img[:, 100:102, 100:105])

import torch.nn as nn
loss_func = nn.BCELoss()
loss = loss_func(torch.tensor([.9]), torch.tensor([.1]))
l2_lambda = .001
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=5,
                       kernel_size=5)
l2_penalty = l2_lambda * sum([(p**2).sum() for p in conv_layer.parameters()])
loss_with_penalty = loss + l2_penalty

linear_layer = nn.Linear(in_features=10, out_features=16)
l2_penalty = l2_lambda * sum([(p**2).sum() for p in linear_layer.parameters()])
loss_with_penalty = loss + l2_penalty

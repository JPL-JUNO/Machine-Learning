"""
@Description: Writing custom layers in PyTorch
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-12 22:42:10
"""
import torch
import torch.nn as nn
from torch import Tensor


class NoisyLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, noise_stddev: float = .1):
        super().__init__()
        w = torch.Tensor(input_size, output_size)
        # nn.Parameter is a Tensor
        # that's a module parameter
        self.w = nn.Parameter(w)
        nn.init.xavier_uniform_(self.w)
        b = torch.Tensor(output_size).fill_(0)
        self.b = nn.Parameter(b)
        self.noise_stddev = noise_stddev

    def forward(self, x: Tensor, training: bool = False):
        if training:
            noise = torch.normal(.0, self.noise_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            # 在推理inference或者评估evaluation是不使用噪声项
            x_new = x
        return torch.add(torch.mm(x_new, self.w), self.b)


if __name__ == "__main__":
    torch.manual_seed(1)
    noisy_layer = NoisyLinear(4, 2)
    x = torch.zeros(size=(1, 4))
    # 直接调用前向传播forward
    print(noisy_layer(x, training=True))
    print(noisy_layer(x, training=True))
    print(noisy_layer(x, training=False))

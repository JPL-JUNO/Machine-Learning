"""
@Description: Pytorch's computation graphs
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-11 11:34:43
"""

import torch


def compute_z(a, b, c):
    """计算2*(a-b)+c"""
    r1 = torch.sub(a, b)
    r2 = torch.mul(r1, 2)
    z = torch.add(r2, c)
    return z


print('Scaler inputs:', compute_z(
    torch.tensor(1), torch.tensor(2), torch.tensor(3)))
print('Rank 1 Inputs:', compute_z(torch.tensor(
    [1]), torch.tensor([2]), torch.tensor([3])))
print('Rank 1 Inputs:', compute_z(torch.tensor(
    [[1]]), torch.tensor([[2]]), torch.tensor([[3]])))

a = torch.tensor(3.14, requires_grad=True)
print(a)
b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(b)
w = torch.tensor([1.0, 2.0, 3.0])
assert not w.requires_grad
w.requires_grad_()
assert w.requires_grad

import torch.nn as nn
torch.manual_seed(1)
w = torch.empty(2, 3)
nn.init.xavier_normal_(w)
print(w)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.empty(2, 3, requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1, 2, requires_grad=True)
        nn.init.xavier_normal_(self.w2)


w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(.5, requires_grad=True)
x = torch.tensor([1.4])
y = torch.tensor([2.1])
z = torch.add(torch.mul(w, x), b)
loss = (y - z).pow(2).sum()
loss.backward()
print('dL/dW :', w.grad)
print('dL/db :', b.grad)

print(2 * x * ((w * x + b) - y))

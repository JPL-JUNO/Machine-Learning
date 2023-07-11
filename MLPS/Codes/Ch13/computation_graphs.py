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

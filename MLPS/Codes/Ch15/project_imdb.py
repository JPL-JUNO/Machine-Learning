"""
@Description: 基于 IMDb 影评进行情感分析
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-24 21:14:36
"""


from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
train_dataset = IMDB(split='train')
test_dataset = IMDB(split='test')
import torch
torch.manual_seed(1)
train_dataset, valid_dataset = random_split(
    list(train_dataset), [20_000, 5_000])

"""
@Description: Higher-level PyTorch APIs: a short introduction to PyTorch-Lightning
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-13 21:27:10
"""
# BUG
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pkg_resources import parse_version
from torchmetrics import __version__ as torchmetrics_version
from torchmetrics import Accuracy


class MultiLayerPerception(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()

        if parse_version(torchmetrics_version) > parse_version('0.8'):
            self.train_acc = Accuracy(task='multiclass', num_classes=10)
            self.valid_acc = Accuracy(task='multiclass', num_classes=10)
            self.test_acc = Accuracy(task='multiclass', num_classes=10)
        else:
            self.train_acc = Accuracy()
            self.valid_acc = Accuracy()
            self.test_acc = Accuracy()

        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
        all_layers.append(nn.Linear(hidden_units[-1], 10))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        # The forward method implements a simple forward pass
        # that returns the logits (outputs of the last fully
        # connected layer of our network before the softmax layer)
        # when we call our model on the input data.
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_training_epoch_end(self, outs):
        self.log('train_acc', self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_acc', self.valid_acc.compute(), prog_bar=True)
        return loss

    def on_validation_epoch_end(self, outs):
        self.log('valid_acc', self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        return optimizer


from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import os
download_flg = False if os.path.exists('MNIST') else True


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self) -> None:
        MNIST(root=self.data_path, download=download_flg)

    def setup(self, stage=None):
        mnist_all = MNIST(
            root=self.data_path,
            train=True, transform=self.transform, download=download_flg)
        self.train, self.val = random_split(
            mnist_all, [55_000, 5_000], generator=torch.Generator().manual_seed(1))
        self.test = MNIST(
            root=self.data_path, train=False, transform=self.transform, download=download_flg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=4)


torch.manual_seed(1)
mnist_dm = MnistDataModule()
mnist_classifier = MultiLayerPerception()
if torch.cuda.is_available():
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1)
else:
    trainer = pl.Trainer(max_epochs=10)
trainer.fit(model=mnist_classifier, datamodule=mnist_dm)

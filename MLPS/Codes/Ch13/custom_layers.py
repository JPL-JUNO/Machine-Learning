"""
@Description: Writing custom layers in PyTorch
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-13 09:08:44
"""
import torch.nn as nn
import torch
from NoisyLinear import NoisyLinear
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np


class MyNoisyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = NoisyLinear(2, 4, .01)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(4, 4)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(4, 1)
        self.a3 = nn.Sigmoid()

    def forward(self, x, training: bool = False):
        x = self.l1(x, training)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred >= .5).float()


if __name__ == '__main__':
    from implementations_of_common_architectures import train_dl, batch_size, x_valid, y_valid
    num_epochs = 200
    torch.manual_seed(1)
    model = MyNoisyModule()
    print(model)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.015)
    torch.manual_seed(1)
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch, True)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred > .5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()
        loss_hist_train[epoch] /= 100 / batch_size
        accuracy_hist_train[epoch] /= 100 / batch_size
        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred >= 0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(loss_hist_train, lw=4, label='Train loss')
    ax.plot(loss_hist_valid, lw=4, label='Validation loss')
    plt.legend(fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(accuracy_hist_train, lw=4, label='Train acc.')
    ax.plot(accuracy_hist_valid, lw=4, label='Validation acc.')
    plt.legend(fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 3, 3)
    plot_decision_regions(X=x_valid.numpy(),
                          y=y_valid.numpy().astype(np.int_),
                          clf=model)
    ax.set_xlabel(r'$x_1$', size=15)
    ax.xaxis.set_label_coords(1, -0.025)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.yaxis.set_label_coords(-0.025, 1)
    plt.show()

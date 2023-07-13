"""
@Description: Classifying handwritten digits
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-30 15:41:56
"""

from sklearn.datasets import fetch_openml
import sys
sys.path.append('./')
sys.path.append('../')
from utilsML.funcs import minibatch_generator
from utilsML.funcs import mse_loss, accuracy
from modelsNN.NeuralNetMLP import NeuralNetMLP
from utilsML.funcs import compute_mse_and_acc

X, y = fetch_openml('mnist_784', version=1, parser='auto',
                    return_X_y=True)
X = X.values
y = y.astype(int).values
assert X.shape == (70_000, 784)
assert y.shape == (70_000,)
# gradient-based optimization is much more stable under these conditions
X = 2 * (X / 255 - .5)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    img = X[y == i][0].reshape(28, 28)
    ax.imshow(img, cmap='Greys')
axes[0][0].set_xticks([])
axes[1][0].set_yticks([])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    img = X[y == 7][i].reshape(28, 28)
    ax.imshow(img, cmap='Greys')
axes[0][0].set_xticks([])
axes[1][0].set_yticks([])
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10_000,
                                                  random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5_000, random_state=123, stratify=y_temp)

import numpy as np
num_epochs = 50
minibatch_size = 100
for i in range(num_epochs):
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break
assert X_train_mini.shape == (100, 784)
assert y_train_mini.shape == (100,)

model = NeuralNetMLP(num_features=28 * 28,
                     num_hidden=50,
                     num_classes=10)

_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.3f}')
predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.3f}')

mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial validation MSE: {mse:.3f}')
print(f'Initial validation accuracy: {acc*100:.3f}')


def train(model, X_train, y_train,
          X_valid, y_valid, num_epochs: int = 50,
          learning_rate: float = .1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            a_h, a_out = model.forward(X_train_mini)

            d_loss__d_w_out, d_loss__d_b_out, \
                d_loss__d_w_h, d_loss__d_b_h = model.backward(
                    X_train_mini, a_h, a_out, y_train_mini)
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        _, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(
            f'Epoch: {e+1:03d}/{num_epochs:03d} '
            f'| Train MSE: {train_mse:.2f} '
            f'| Train Acc: {train_acc:.2f}% '
            f'| Valid Acc: {valid_acc:.2f}%')
    return epoch_loss, epoch_train_acc, epoch_valid_acc


epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid, num_epochs=50, learning_rate=.1)

plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.show()

plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]
_, probas = model.forward(X_test_subset)
test_pred = np.argmax(probas, axis=1)
mask = y_test_subset != test_pred
misclassified_images = X_test_subset[mask][:25]
misclassified_labels = test_pred[mask][:25]
correct_labels = y_test_subset[mask][:25]
fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(8, 8))
for i, ax in enumerate(axes.ravel()):
    img = misclassified_images[i].reshape(28, 28)
    ax.imshow(img, cmap='Greys', interpolation='nearest')
    ax.set_title(f'{i+1}) True: {correct_labels[i]}\n'
                 f'Predicted: {misclassified_labels[i]}')
axes[0][0].set_xticks([])
axes[1][0].set_xticks([])
plt.tight_layout()
plt.show()

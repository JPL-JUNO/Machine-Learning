"""
@Description: Getting started with the logistic function
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 21:22:02
"""

import numpy as np
from numpy import ndarray


def sigmoid(input: ndarray) -> ndarray:
    return 1.0 / (1 + np.exp(-input))


z = np.linspace(-8, 8, 1000)
y = sigmoid(z)
import matplotlib.pyplot as plt
plt.plot(z, y)
plt.axhline(y=0, ls='dotted', color='k')
plt.axhline(y=.5, ls='dotted', color='k')
plt.axhline(y=1, ls='dotted', color='k')
plt.yticks([0, .25, .5, .75, 1.0])
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.show()

y_hat = np.linspace(0.0001, 0.9999, 1000)
cost = -np.log(y_hat)
fig = plt.figure()
plt.plot(y_hat, cost, label=r'For ground truth $y_i=1$')
plt.xlabel('Prediction')
plt.ylabel('Cost')
plt.axis([0, 1.0, 0, 7])
cost = -np.log(1 - y_hat)
plt.plot(y_hat, cost, label=r'For ground truth $y_i=0$')
plt.legend()
plt.show()

"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-01 23:41:13
"""

import numpy as np
from numpy import ndarray
import sys
sys.path.append('./')
sys.path.append('../')
from utilsML.funcs import sigmoid
from utilsML.funcs import int_to_onehot


class NeuralNetMLP:
    def __init__(self, num_features: int, num_hidden: int,
                 num_classes: int, random_state: int = 123):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_state)

        # hidden layer
        # 使用转置的方法相比于没转置效果要差一些
        self.weight_h = rng.normal(
            loc=0.0, scale=.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x) -> tuple[ndarray]:
        """return the hidden and output layer activations

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        tuple[ndarray]
            (hidden activations, output activations)
        """
        # hidden layer
        # input dim : [n_samples, n_features]
        # output dim : [n_samples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # output layer
        # input dim : [n_samples, n_hidden]
        # output dim : [n_samples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        """calculates the gradients of the loss with respect to the weights and bias parameters

        Parameters
        ----------
        x : _type_
            _description_
        a_h : _type_
            _description_
        a_out : _type_
            _description_
        y : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        y_onehot = int_to_onehot(y, self.num_classes)

        # part1: dLoss/dOutWeights
        # = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeights
        # where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        # for convenient re-use

        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]

        d_a_out__d_z_net = a_out * (1 - a_out)

        delta_out = d_loss__d_a_out * d_a_out__d_z_net

        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # part 2: dLoss/dHiddenWeights
        # = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenAct

        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = a_h * (1 - a_h)
        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T,
                               d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)

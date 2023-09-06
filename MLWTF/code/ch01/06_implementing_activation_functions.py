"""
@Title: Implementing activation functions
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 16:06:24
@Description: 
"""
import tensorflow as tf
from tensorflow import nn

print(tf.nn.relu([-3., 3., 10.]))

# There are times where we'll want to cap the linearly increasing part of the preceding
# ReLU activation function. We can do this by nesting the max(0,x) function in a min()
# function. The implementation that TensorFlow has is called the ReLU6 function. This
# is defined as min(max(0,x),6). This is a version of the hard-sigmoid function, is
# computationally faster, and does not suffer from vanishing (infinitesimally near zero)
# or exploding values.
print(tf.nn.relu6([-3., 3., 10.]))


# The sigmoid function is the most common continuous and smooth activation
# function. It is also called a logistic function and has the form 1 / (1 + exp(-x)). The
# sigmoid function is not used very often because of its tendency to zero-out the
# backpropagation terms during training.
print(tf.nn.sigmoid([-10., 0., 1.]))
# tf.Tensor([0.26894143 0.5        0.9999546 ], shape=(3,), dtype=float32)


# Another smooth activation function is the hyper tangent. The hyper tangent function
# is very similar to the sigmoid except that instead of having a range between 0 and
# 1, it has a range between -1 and 1. This function has the form of the ratio of the
# hyperbolic sine over the hyperbolic cosine.
print(tf.nn.tanh([-1., 0., 1.]))
# tf.Tensor([-0.7615942  0.         0.7615942], shape=(3,), dtype=float32)


# The softsign function is also used as an activation function. The form of this
# function is x/(|x| + 1). The softsign function is supposed to be a continuous (but
# not smooth) approximation to the sign function.
print(tf.nn.softsign([-1., 0., 1.]))
# tf.Tensor([-0.5  0.   0.5], shape=(3,), dtype=float32)


# Another function, the softplus function, is a smooth version of the ReLU function.
# The form of this function is log(exp(x) + 1).
print(tf.nn.softplus([-1., 0., 1.]))
# tf.Tensor([0.3132617 0.6931472 1.3132616], shape=(3,), dtype=float32)


# The Exponential Linear Unit (ELU) is very similar to the softplus function except that
# the bottom asymptote is -1 instead of 0. The form is (exp(x) + 1) if x < 0, else x.
print(tf.nn.elu([-1., 0., 1.]))
# tf.Tensor([-0.63212055  0.          1.        ], shape=(3,), dtype=float32)


def swish(x):
    # https://arxiv.org/abs/1710.05941
    return x * tf.nn.sigmoid(x)


print(swish([-1., 0., 1.]))
# tf.Tensor([-0.26894143  0.          0.7310586 ], shape=(3,), dtype=float32)

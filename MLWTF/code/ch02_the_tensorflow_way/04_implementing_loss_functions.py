"""
@Title: Implementing loss functions
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-07 21:35:53
@Description: 
"""
import matplotlib.pyplot as plt
import tensorflow as tf

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# 以下是回归问题的损失函数


def l2_norm(y_true, y_pred):
    """计算 L2 范数"""
    # tf.nn.l2_loss() = .5 * l2_norm()
    return tf.square(y_true - y_pred)


def l1_norm(y_true, y_pred):
    """计算 L1 范数"""
    # 对异常值比 L2 更好，
    # 但是 not smooth that can result in algorithms not converging well
    return tf.abs(y_true - y_pred)


def phuber1(y_true, y_pred):
    delta1 = tf.constant(.25)
    return tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((y_true - y_pred) / delta1) - 1.))


def phuber2(y_true, y_pred):
    delta2 = tf.constant(5.)
    return tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((y_true - y_pred) / delta2) - 1.))


x_vals = tf.linspace(-3., 5., 500)
target - tf.fill([500,], 1.)


def hinge(y_true, y_pred):
    """
    Hinge loss is mostly used for support vector machines but can be used in neural networks as
    well. It is meant to compute a loss among two target classes, 1 and -1.
    """
    return tf.maximum(0., 1. - tf.multiply(y_true, y_pred))


def x_entropy(y_true, y_pred):
    """
    Cross-entropy loss for a binary case is also sometimes referred to as the logistic loss
    function.

    `y_pred` 一般是 (0, 1) 实值数
    """
    return -(tf.multiply(y_true, tf.math.log(y_pred)) + tf.multiply((1 - y_true), tf.math.log(1. - y_pred)))


def x_entropy_sigmoid(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)


def x_entropy_weighted(y_true, y_pred):
    weight = tf.constant(.5)
    return tf.nn.weighted_cross_entropy_with_logits(labels=y_true,
                                                    logits=y_pred,
                                                    pos_weight=weight)


def softmax_x_entropy(y_true, y_pred):
    """
    Softmax cross-entropy loss operates on non-normalized outputs. This function is used to
    measure a loss when there is only one target category instead of multiple.
    >>> unscaled_logits = tf.constant([[1., -3., 10.]])
    >>> target_dist = tf.constant([[0.1, 0.02, 0.88]])
    >>> softmax_x_entropy(y_true=target_dist,
                          y_pred=unscaled_logits)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([1.1601256], dtype=float32)>
    >>> # 应该等价于下面的这条代码
    >>> -tf.reduce_sum(tf.multiply(target_dist, tf.math.log(tf.math.exp(unscaled_logits)/tf.reduce_sum(tf.math.exp(unscaled_logits)))))
    """
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)


def sparse_x_entropy(y_true, y_pred):
    """
    Sparse softmax cross-entropy loss is almost the same as softmax cross-entropy loss, except instead of the target being a probability distribution, it is an index of which category is true.

    >>> unscaled_logits = tf.constant([[1., -3., 10.]])
    >>> target_dist = tf.constant([2])
    >>> sparse_x_entropy(y_true=target_dist,
                         y_pred=unscaled_logits)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.00012564], dtype=float32)>
    >>> # 似乎等价于下面这条计算逻辑
    >>> -tf.reduce_max(tf.math.log(tf.math.exp(unscaled_logits)/tf.reduce_sum(tf.math.exp(unscaled_logits))))
    >>> 
    """
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)


x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)
funcs = [(l2_norm, "b-", "L2 Norm Loss"),
         (l1_norm, "r--", "L1 Norm Loss"),
         (phuber1, "k-.", "P-Huber Loss(.25)"),
         (phuber2, "g:", "P-Huber Loss(5.0)")]

for func, line_type, func_name in funcs:
    plt.plot(x_vals, func(y_true=target, y_pred=x_vals),
             line_type, label=func_name)
plt.ylim(-0.2, 5)
plt.title("Regression loss functions")
plt.legend(prop={'size': 11})
plt.show()

x_vals = tf.linspace(-3., 5., 500)
target = tf.fill([500,], 1.)
funcs = [(hinge, 'b-', "Hinge Loss"),
         (x_entropy, 'r--', "Cross entropy loss"),
         (x_entropy_sigmoid, "k-.", "Cross entropy sigmoid loss"),
         (x_entropy_weighted, 'g:', "Weighted cross entropy loss(x=.5)"),]
for func, line_type, func_name in funcs:
    plt.plot(x_vals, func(y_true=target, y_pred=x_vals),
             line_type, label=func_name)
plt.ylim(-1.5, 3)
plt.title("Classification loss functions")
plt.legend(prop={'size': 11})
plt.show()

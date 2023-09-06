"""
@Title: Layering nested operations 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 22:04:18
@Description: 
"""

import tensorflow as tf
import numpy as np

my_array = np.array([[1., 3., 5., 7., 9.],
                     [-2., 0., 2., 4., 6.],
                     [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.Variable(x_vals, dtype=tf.float32)

m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])


def prod1(a, b):
    return tf.matmul(a, b)


def prod2(a, b):
    return tf.matmul(a, b)


def add1(a, b):
    return tf.add(a, b)


result = add1(prod2(prod1(x_data, m1), m2), a1)
print(result.numpy())


class Operations:
    def __init__(self, a):
        self.result = a

    def apply(self, func, b):
        self.result = func(self.result, b)
        return self


operation = Operations(a=x_data).apply(
    prod1, b=m1).apply(prod2, b=m2).apply(add1, a1)
print(operation.result.numpy())

# 🚩
# This is not always the
# case. There may be a dimension or two that we do not know beforehand or some that can
# vary during our data processing. To take this into account, we designate the dimension or
# dimensions that can vary (or are unknown) as value None.
v = tf.Variable(initial_value=tf.random.normal(shape=(1, 5)),
                shape=tf.TensorShape((None, 5)))
# v 可以随时改变，不需要确定值
v.assign(tf.random.normal(shape=(10, 5)))
# It is fine for matrix multiplication to have flexible rows because that won't affect the
# arrangement of our operations. This will come in handy in later chapters when we are feeding
# data in multiple batches of varying batch sizes.
# 尽管可以，但是如果可以事先确定维度大小，应该指定

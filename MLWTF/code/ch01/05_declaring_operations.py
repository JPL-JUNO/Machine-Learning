"""
@Title: Declaring operations
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 15:20:57
@Description: 
"""
import tensorflow as tf

print(tf.math.divide(3, 4))
print(tf.math.truediv(3, 4))


# If we have floats and want integer division, we can use the floordiv() function.
# Note that this will still return a float, but it will be rounded down to the nearest
# integer.
print(tf.math.floordiv(3.0, 4.0))

# Another important function is mod().
# This function returns the remainder after division.
print(tf.math.mod(22., 5.))


# 🚩
# 叉积
# The cross product between two tensors is achieved by the cross() function.
# Remember that the cross product is only defined for two three-dimensional vectors,
# so it only accepts two three-dimensional tensors.
print(tf.linalg.cross([1., 0., 0.], [0., 1., 0.]))

print(tf.math.abs([-2, -4, -9]))
print(tf.math.ceil([1.1, 1.05, 2.2]))
print(tf.math.cos([1.1, 1.05, 2.2]))
print(tf.math.exp([1.1, 1.05, 2.2]))
print(tf.math.floor([-1.1, 1.05, 2.2]))
print(tf.math.log([1.1, 1.05, 2.2]))
print(tf.math.maximum([1.1, 1.05, 2.2], [2.2, 1., 3.4]))
print(tf.math.negative([1.1, 1.05, 2.2]))
print(tf.math.pow([1.1, 1.05, 2.2], [2, 3, 4]))


def custom_polynomial(value):
    return tf.math.subtract(3 * tf.math.square(value), value) + 10


print(custom_polynomial(11))

# 在自定义函数之前，最好先看一些文档，避免重复造轮子

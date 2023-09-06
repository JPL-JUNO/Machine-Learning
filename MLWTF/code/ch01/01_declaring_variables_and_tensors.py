"""
@Title: declaring variables and tensors
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 09:16:50
@Description: 声明变量与张量
"""

import tensorflow as tf

# 1. Fixed size tensors:
row_dim, col_dim = 3, 3
zero_tsr = tf.zeros(shape=[row_dim, col_dim], dtype=tf.float32)

ones_tsr = tf.ones(shape=[row_dim, col_dim])

filled_tsr = tf.fill([row_dim, col_dim], value=42)

constant_tsr = tf.constant([1, 2, 3])
# Note that the tf.constant() function can be used to
# broadcast a value into an array, mimicking the behavior of
# tf.fill() by writing tf.constant(42, [row_dim, col_
# dim]).

# 2. Tensors of similar shape:
zeros_similar = tf.zeros_like(constant_tsr)

ones_similar = tf.ones_like(constant_tsr)

# Note that since these tensors depend on prior tensors, we
# must initialize them in order. Attempting to initialize the
# tensors in a random order will result in an error.

# 3. Sequence tensors:

linear_tsr = tf.linspace(start=0., stop=1., num=3)
# Note that the start and stop parameters should be float
# values, and that num should be an integer.
# this function includes the specified stop value.

integer_seq_tsr = tf.range(start=6, limit=15, delta=3)
# Note that this function does not include the
# limit value and it can operate with both integer and float values for the start and limit
# parameters.

# 4. Random tensors:
randunif_tsr = tf.random.uniform([row_dim, col_dim], minval=0, maxval=1)
# [minval, maxval)
randnorm_tsr = tf.random.normal([row_dim, col_dim], mean=0., stddev=1.)


# The truncated_normal() function always picks normal values within
# two standard deviations of the specified mean:
runcnorm_tsr = tf.random.truncated_normal(
    (row_dim, col_dim), mean=0., stddev=1.)


shuffled_output = tf.random.shuffle(constant_tsr)
cropped_output = tf.image.random_crop(runcnorm_tsr, size=[2, 2])


height, width = (64, 64)
my_image = tf.random.uniform((height, width, 3),
                             minval=0, maxval=255, dtype=tf.int32)
cropped_image = tf.image.random_crop(
    my_image, size=[height // 2, width // 2, 3])


# create the corresponding variables by wrapping the tensor in the Variable() function
my_var = tf.Variable(tf.zeros((row_dim, col_dim)))

"""
@Title: Working with multiple layers
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-07 20:43:37
@Description: 
"""

import tensorflow as tf

batch_size = [1]
x_shape = [4, 4, 1]
x_data = tf.random.uniform(shape=batch_size + x_shape)


def mov_avg_layer(x):
    """这里维度方向没太理解，尤其是这个过滤器的shape"""
    my_filter = tf.constant(.25, shape=[2, 2, 1, 1])
    my_strides = [1, 2, 2, 1]
    layer = tf.nn.conv2d(x, my_filter, my_strides,
                         padding="SAME", name="Moving_Avg_Window")
    return layer


def custom_layer(input_matrix):
    input_matrix_squeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    B = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_squeezed)
    temp2 = tf.add(temp1, B)
    return tf.sigmoid(temp2)


first_layer = mov_avg_layer(x_data)
second_layer = custom_layer(first_layer)

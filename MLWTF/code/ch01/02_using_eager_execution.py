"""
@Title: Using eager execution
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 09:48:51
@Description: 
"""
import tensorflow as tf
assert tf.executing_eagerly()

x = [[2.]]
m = tf.matmul(x, x)
print("the result is {}".format(m))
# As TensorFlow is now set on eager execution as default, you won't be surprised to hear
# that tf.Session has been removed from the TensorFlow API.

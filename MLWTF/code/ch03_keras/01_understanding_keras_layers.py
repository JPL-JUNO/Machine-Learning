"""
@Title: Understanding Keras layers
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-08 20:36:30
@Description: 此文件不能运行，仅展示作用
"""

import tensorflow as tf
from tensorflow import keras

# The get_weights() function returns the weights of the layer as a list of NumPy arrays:
layer.get_weights()
# The set_weights() method fixes the weights of the layer from a list of Numpy arrays:
layer.set_weights(weights)

# We can easily get the inputs and outputs of a layer by using this
# command if the layer is a single node (no shared layer):
# 有些层的参数是共享的，如果不是共享的可以如下
layer.input
layer.output

# Or this one, if the layer has multiple nodes:
layer.get_input_at(node_index)
layer.get_output_at(node_index)

# We can also easily get the layer's input and output shapes by using this command
# if a layer is a single node (no shared layer):
layer.input_shape
layer.output_shape

layer.get_input_shape_at(node_index)
layer.get_output_shape_at(node_index)

# The get_config() function returns a dictionary containing
# the configuration of the layer:
layer.get_config()

# The from_config() method instantiates a layer's configuration:
lay_from_config(config)

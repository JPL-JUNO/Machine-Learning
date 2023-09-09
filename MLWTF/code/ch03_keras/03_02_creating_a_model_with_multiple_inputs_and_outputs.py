"""
@Title: Creating a model with multiple inputs and outputs
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-09 15:30:53
@Description: 
"""
# The model will have two inputs:
# 1. Data about the house such as the number of bedrooms, house size, air
# conditioning, fitted kitchen, etc.
# 2. A recent picture of the house

# This model will have two outputs:
# 1. The elapsed time before the sale (two categories – slow or fast)
# 2. The predicted price


import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
import keras.models

# start by building the first block to process tabular data
# about the house.
house_data_inputs = tf.keras.Input(shape=(128,), name="house_data")
x = tf.keras.layers.Dense(64, activation="relu")(house_data_inputs)
block_1_output = tf.keras.layers.Dense(32, activation="relu")(x)

# build the second block to process the house image data
house_picture_inputs = tf.keras.Input(
    shape=(128, 128, 3), name="house_picture")
x = tf.keras.layers.Conv2D(64, 3, activation="relu",
                           padding="same")(house_picture_inputs)
x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = tf.keras.layers.Flatten()(x)

# merge all available features into a single large vector via concatenation.
x = tf.keras.layers.concatenate([block_1_output, block_2_output])


price_pred = tf.keras.layers.Dense(1, name="price", activation="relu")(x)
time_elapsed_pred = tf.keras.layers.Dense(
    2, name="elapsed_time", activation="softmax")(x)

model = keras.Model([house_data_inputs, house_picture_inputs],
                    [price_pred, time_elapsed_pred],
                    name="toy_house_pred")
keras.utils.plot_model(model, "multi_input_and_output_model.png",
                       show_shapes=True)

# Shared layers

# Variable-length sequence of integers
text_input_a = tf.keras.Input(shape=(None, ), dtype="int32")
text_input_b = tf.keras.Input(shape=(None, ), dtype="int32")

# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = tf.keras.layers.Embedding(1_000, 128)

# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)


# Extracting and reusing nodes in the graph of layers

resnet = tf.keras.applications.resnet.ResNet50()

intermediate_layers = [layer.output for layer in resnet.layers]

intermediate_layers[:10]

feature_layers = intermediate_layers[:-2]

feat_extraction_model = keras.Model(
    inputs=resnet.input, outputs=feature_layers)

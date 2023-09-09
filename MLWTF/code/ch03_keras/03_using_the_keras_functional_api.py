"""
@Title: Using the Keras Functional API
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-09 14:24:44
@Description: 
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
import keras.models

# load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_mnist_train, y_mnist_train), (X_mnist_test, y_mnist_test) = mnist.load_data()

# Remember that
# in Keras, the input layer is not a layer but a tensor, and we have to specify the input
# shape for the first layer. This tensor must have the same shape as our training data.
inputs = tf.keras.Input(shape=(28, 28))

# we will flatten the images of size (28,28) using the following command.
flatten_layer = keras.layers.Flatten()

# We'll add a new node in the graph of layers by calling the flatten_layer on the
# inputs object:
flatten_output = flatten_layer(inputs)

# we'll create a new layer instance:
dense_layer = tf.keras.layers.Dense(50, activation="relu")

# We'll add a new node:
dense_output = dense_layer(flatten_output)

# we will add another
# dense layer to do a classification task between 10 classes:
predictions = tf.keras.layers.Dense(10, activation="softmax")(dense_output)

# Input tensor(s) and output tensor(s) are used to define a model. The model is a
# function of one or more input layers and one or more output layers. The model
# instance formalizes the computational graph on how the data flows from input(s) to
# output(s).
model = keras.Model(inputs=inputs, outputs=predictions)

print(model.summary())

model.compile(optimizer="sgd",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X_mnist_train, y_mnist_train,
          validation_data=(X_mnist_test, y_mnist_test),
          epochs=10)

x = Input(shape=(784, ))
y = model(x)

# Input tensor for sequences of 10 timesteps,
# Each containing a 28x28 dimensional matrix.
input_sequences = tf.keras.Input(shape=(10, 28, 28))
# We will apply the previous model to each sequence so one for each
#  timestep.
# The MNIST model returns a vector with 10 probabilities (one for
# each digit).
# The TimeDistributed output will be a sequence of 50 vectors of
# size 10.
processed_sequences = tf.keras.layers.TimeDistributed(model)(input_sequences)

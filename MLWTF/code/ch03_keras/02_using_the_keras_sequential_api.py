"""
@Title: Using the Keras Sequential API
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-08 20:49:12
@Description: 
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np

# passing a list of layer instances as an array to the constructor.
model = tf.keras.Sequential([
    # Add a fully connected layer with 1024 units to the model
    tf.keras.layers.Dense(1024, input_dim=64),
    # Add an activation layer with ReLU activation function
    tf.keras.layers.Activation("relu"),
    # Add a fully connected layer with 256 units to the model
    tf.keras.layers.Dense(256),
    # Add an activation layer with ReLU activation function
    tf.keras.layers.Activation("relu"),
    # Add a fully connected layer with 10 units to the model
    tf.keras.layers.Dense(10),
    # Add an activation layer with softmax activation function
    tf.keras.layers.Activation("softmax"),
])

# Another way to create a Sequential model is to instantiate a Sequential class and
# then add layers via the .add() method.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1024, input_dim=64))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation("softmax"))

# We can add an activation function by specifying the name of a built-in
# function or as a callable object.
tf.keras.layers.Dense(256, activation="sigmoid")
tf.keras.layers.Dense(256, activation=tf.keras.activations.sigmoid)


# We can also specify an initialization strategy for the initial weights (kernel
# and bias) by passing the string identifier of built-in initializers or a callable
# object. The kernel is by default set to the "Glorot uniform" initializer, and the
# bias is set to zeros.
tf.keras.layers.Dense(256, kernel_initializer="random_normal")
tf.keras.layers.Dense(
    256, bias_initializer=tf.keras.initializers.Constant(value=5))

# We can also specify regularizers for kernel and bias, such as L1 (also called
# Lasso) or L2 (also called Ridge) regularization. By default, no regularization
# is applied.
tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l1(0.01))
tf.keras.layers.Dense(256, bias_regularizer=tf.keras.regularizers.l2(0.01))

# In Keras, it's strongly recommended to set the input shape for the first layer.
tf.keras.layers.Dense(256, input_dim=(64))
tf.keras.layers.Dense(256, input_dim=(64), batch_size=10)

# Before the learning phase, our model needs to be configured. This is done by the
# compile method. We have to specify:
# 1. 优化算法 An optimization algorithm for the training of our neural network.
# 2. 损失函数 A loss function called an objective function or optimization score function
# aims at minimizing the model.
# 3. 评价指标 A list of metrics used to judge our model's performance that aren't used in
# the model training process.
# 4. If you want to be sure that the model trains and evaluates eagerly, we can
# set the argument run_eagerly to true.

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# three toy datasets of 64 features with random values.
data = np.random.random((2_000, 64))
labels = np.random.random((2_000, 10))
val_data = np.random.random((500, 64))
val_labels = np.random.random((500, 10))
test_data = np.random.random((500, 64))
test_labels = np.random.random((500, 10))

model.fit(data, labels, epochs=10, batch_size=50,
          validation_data=(val_data, val_labels))

model.evaluate(data, labels)
# 应该是下面这一行进行评估
# model.evaluate(test_data, test_labels)

# 进行预测
result = model.predict(data, batch_size=50)

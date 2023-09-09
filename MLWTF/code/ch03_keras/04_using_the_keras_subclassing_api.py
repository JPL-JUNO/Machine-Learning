"""
@Title: Using the Keras Subclassing API
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-09 15:58:59
@Description: 
"""

import tensorflow as tf
from tensorflow import keras

# Creating a custom layer


# All layers are subclasses of the Layer class and implement these methods:

# 1. The build method, which defines the weights of the layer.
# 2. The call method, which specifies the transformation from inputs to outputs done by the layer.
# 3. The compute_output_shape method, if the layer modifies the shape of its input. This allows Keras to perform automatic shape inference.
# 4. The get_config and from_config methods, if the layer is serialized and deserialized.

class MyCustomerDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MyCustomerDense, self).__init__()
        self.units = units

    # Define the weights and the bias
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units, ),
                                 initializer="random_normal",
                                 trainable=True)

    # Applying this layer transformation to the input tensor
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


inputs = keras.Input(shape=(12, 4))
# 调用 call 方法
outputs = MyCustomerDense(2)(inputs)
# 通过输入，输出建立模型
model = keras.Model(inputs, outputs)
config = model.get_config()

# we will reload the model from the config:
# 可以从配置中建立新的模型
# custom_objects的 key 要和 class name 一样
new_model = keras.Model.from_config(
    config, custom_objects={"MyCustomerDense": MyCustomerDense})


mnist = tf.keras.datasets.mnist
(X_mnist_train, y_mnist_train), (X_mnist_test, y_mnist_test) = mnist.load_data()
train_mnist_features = X_mnist_train / 255
test_mnist_features = X_mnist_test / 255


class MyMnistModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(MyMnistModel, self).__init__(name="my_mnist_model")
        self.num_classes = num_classes

        self.flatten_1 = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(.1)
        self.dense_1 = tf.keras.layers.Dense(50, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.flatten_1(inputs)

        x = self.dense_1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense_2(x)


my_mnist_model = MyMnistModel(10)
my_mnist_model.compile(optimizer="sgd",
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
my_mnist_model.fit(train_mnist_features, y_mnist_train,
                   validation_data=(test_mnist_features, y_mnist_test),
                   epochs=10)

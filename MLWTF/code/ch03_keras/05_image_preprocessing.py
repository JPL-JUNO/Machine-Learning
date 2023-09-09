"""
@Title: Image preprocessing
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-10 00:58:35
@Description: 
"""

import tensorflow as tf

(X_cifar10_train, y_cifar10_train), (X_cifar10_test,
                                     y_cifar10_test) = tf.keras.datasets.cifar10.load_data()
X_cifar10_train = X_cifar10_train[:30_000]
y_cifar10_train = y_cifar10_train[:30_000]

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    width_shift_range=3,
    height_shift_range=3,
    horizontal_flip=True)


it = datagen.flow(X_cifar10_train, y_cifar10_train, batch_size=32)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same",
                           activation="relu", input_shape=[32, 32, 3]),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                           padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                           padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                           padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=.01),
              metrics=["accuracy"])
history = model.fit(it,
                    epochs=10,
                    steps_per_epoch=len(X_cifar10_train) / 32,
                    validation_data=(X_cifar10_test, y_cifar10_test))

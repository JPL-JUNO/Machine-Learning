"""
@Title: Combining everything together
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-08 10:42:24
@Description: 
"""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

batch_size = 20

# 保留10%作为测试，但是我不确定是否是最后的10%（因为可能都是相同标签的样本）
# parameter as_supervised=True, that will allow us to access the data as
# tuples of features and labels when iterating from the dataset.
iris = tfds.load("iris", split="train[:90%]", as_supervised=True)
iris_test = tfds.load("iris", split="train[90%:]", as_supervised=True)


def iris2d(features, label):
    return features[2:], tf.cast((label == 0), dtype=tf.float32)


# 生成器可以动态的投喂数据，而不需要一次性加载全部数据进内存
train_generator = iris.map(iris2d).shuffle(buffer_size=100).batch(batch_size)
test_generator = iris_test.map(iris2d).batch(1)


def linear_model(X, A, b):
    my_output = tf.add(tf.matmul(X, A), b)
    return tf.squeeze(my_output)


def x_entropy(y_true, y_pred):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                logits=y_pred)
    )


my_opt = tf.optimizers.SGD(learning_rate=.02)
tf.random.set_seed(1)
np.random.seed(0)
A = tf.Variable(tf.random.normal(shape=[2, 1]))
b = tf.Variable(tf.random.normal(shape=[1]))
history = list()

for i in range(300):
    iteration_loss = list()
    for features, label in train_generator:
        with tf.GradientTape() as tape:
            predictions = linear_model(features, A, b)
            loss = x_entropy(label, predictions)
        iteration_loss.append(loss.numpy())
        gradients = tape.gradient(loss, [A, b])
        my_opt.apply_gradients(zip(gradients, [A, b]))
    history.append(np.mean(iteration_loss))
    if (i + 1) % 30 == 0:
        print(f"Step #{i+1} Weights: {A.numpy().T}\
            Biases: {b.numpy()}")
        print(f"Loss = {loss.numpy()}")

plt.plot(history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()

predictions = list()
labels = list()
for features, label in test_generator:
    predictions.append(linear_model(features, A, b).numpy())
    labels.append(label.numpy()[0])
test_loss = x_entropy(np.array(labels), np.array(predictions)).numpy()
print(f"test cross-entropy is {test_loss}")

coefficients = np.ravel(A.numpy())
intercept = b.numpy()

for j, (features, label) in enumerate(train_generator):
    setosa_mask = label.numpy() == 1
    setosa = features.numpy()[setosa_mask]
    non_setosa = features.numpy()[~setosa_mask]
    plt.scatter(setosa[:, 0], setosa[:, 1], c='red', label='setosa')
    plt.scatter(non_setosa[:, 0], non_setosa[:, 1],
                c="blue", label="non-setosa")
    if j == 0:
        plt.legend()
a = -coefficients[0] / coefficients[1]
xx = np.linspace(plt.xlim()[0], plt.xlim()[1], num=10_000)
yy = a * xx - intercept / coefficients[1]
on_the_plot = (yy > plt.ylim()[0]) & (yy < plt.ylim()[1])
plt.plot(xx[on_the_plot], yy[on_the_plot], 'k--')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

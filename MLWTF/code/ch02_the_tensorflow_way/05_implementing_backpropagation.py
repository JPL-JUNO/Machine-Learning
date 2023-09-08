"""
@Title: Implementing backpropagation
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-08 09:21:59
@Description: 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 演示一个回归的问题
np.random.seed(0)
x_vals = np.random.normal(1, .1, 100).astype(np.float32)
y_vals = (x_vals * (np.random.normal(1, .05, 100) - .5)).astype(np.float32)
plt.scatter(x_vals, y_vals)
plt.show()


def my_output(X, weights, biases):
    # tf.add(tf.matmul(X, weights), biases)
    return tf.add(tf.multiply(X, weights), biases)


def loss_func(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))


my_opt = tf.optimizers.SGD(learning_rate=.02)

tf.random.set_seed(1)
np.random.seed(0)
weights = tf.Variable(tf.random.normal(shape=[1]))
biases = tf.Variable(tf.random.normal(shape=[1]))
# set a recording list (named history) to help us visualize the optimization steps:
history = list()

for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    with tf.GradientTape() as tape:
        # 上下文的变量都会被监控，常量不会被监控
        # 除非 tape.watch(constant)
        predictions = my_output(rand_x, weights, biases)
        loss = loss_func(rand_y, predictions)
    history.append(loss.numpy())
    gradients = tape.gradient(loss, [weights, biases])
    my_opt.apply_gradients(zip(gradients, [weights, biases]))
    if (i + 1) % 25 == 0:
        print(
            f"Step #{i+1} Weights: {weights.numpy()} Biases: {biases.numpy()}")
        print(f"Loss = {loss.numpy()}")
plt.plot(history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()

# 演示一个分类问题
np.random.seed(0)
x_vals = np.concatenate((np.random.normal(-3, 1, 50),
                         np.random.normal(3, 1, 50))).astype(np.float32)
y_vals = np.concatenate((np.repeat(0., 50),
                         np.repeat(1., 50))).astype(np.float32)
plt.hist(x_vals[y_vals == 1], color='b')
plt.hist(x_vals[y_vals == 0], color='r')
plt.show()


def loss_func(y_true, y_pred):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                logits=y_pred)
    )


# 优化器需要再次设置？不然报错
my_opt = tf.optimizers.SGD(learning_rate=.02)

tf.random.set_seed(1)
np.random.seed(0)
weights = tf.Variable(tf.random.normal(shape=[1]))
biases = tf.Variable(tf.random.normal(shape=[1]))
history = list()

for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    with tf.GradientTape() as tape:
        predictions = my_output(rand_x, weights, biases)
        loss = loss_func(rand_y, predictions)
    history.append(loss.numpy())
    gradients = tape.gradient(loss, [weights, biases])
    my_opt.apply_gradients(zip(gradients, [weights, biases]))
    if (i + 1) % 25 == 0:
        print(
            f"Step #{i+1} Weights: {weights.numpy()} Biases: {biases.numpy()}")
        print(f"Loss = {loss.numpy()}")
plt.plot(history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()

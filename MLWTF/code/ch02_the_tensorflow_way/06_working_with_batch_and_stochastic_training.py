"""
@Title: Working with batch and stochastic training
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-08 10:41:41
@Description: 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

batch_size = 20
np.random.seed(0)
x_vals = np.random.normal(1, .1, 100).astype(np.float32)
y_vals = (x_vals * (np.random.normal(1, .05, 100) - .5)).astype(np.float32)


def loss_func(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))


def my_output(X, weights, biases):
    # tf.add(tf.matmul(X, weights), biases)
    return tf.add(tf.multiply(X, weights), biases)


tf.random.set_seed(1)
np.random.seed(0)

history_batch = list()
history_stochastic = list()

for way, batch in [("batch", batch_size), ("stochastic", 1)]:
    my_opt = tf.optimizers.SGD(learning_rate=.02)
    weights = tf.Variable(tf.random.normal(shape=[1]))
    biases = tf.Variable(tf.random.normal(shape=[1]))
    for i in range(50):
        rand_index = np.random.choice(100, size=batch, replace=False)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        with tf.GradientTape() as tape:
            predictions = my_output(rand_x, weights, biases)
            loss = loss_func(rand_y, predictions)
        if way == 'batch':
            history_batch.append(loss.numpy())
        elif way == 'stochastic':
            history_stochastic.append(loss.numpy())
        gradients = tape.gradient(loss, [weights, biases])
        my_opt.apply_gradients(zip(gradients, [weights, biases]))
        if (i + 1) % 25 == 0:
            print(
                f"Step #{i+1} Weights: {weights.numpy()} Biases: {biases.numpy()}")
            print(f"Loss = {loss.numpy()}")
    print()

# batch 从图上来看只会带来损失曲线平滑一些，并不会使得结果更优或者损失更小
plt.plot(history_stochastic, 'b-', label='Stochastic Loss')
plt.plot(history_batch, 'r--', label='Batch Loss')
plt.legend(prop={'size': 11})
plt.show()

# Now our graph displays a smoother trend line. The persistent presence of bumps could be
# solved by reducing the learning rate and adjusting the batch size.

"""
@Title: how tensorflow work
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 10:16:03
@Description:  the general flow of TensorFlow algorithms
"""

# 1. Import or generate datasets:
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

data = tfds.load("iris", split="train")

# 2. Transform and normalize data:
batch_size = 32
for batch in data.batch(batch_size, drop_remainder=True):
    labels = tf.one_hot(batch["label"], 3)
    X = batch["features"]
    X = (X - np.mean(X)) / np.std(X)

# 3. Partition the dataset into training, test, and validation sets:

# 4. Set algorithm parameters (hyperparameters):
epochs = 1_000
batch_size = 32
input_size = 4
output_size = 3
learning_rate = 0.001

# 5. Initialize variables:
weights = tf.Variable(tf.random.normal(shape=(input_size, output_size),
                                       dtype=tf.float32))
biases = tf.Variable(tf.random.normal(shape=(output_size, ),
                                      dtype=tf.float32))

# 6. Define the model structure:
logits = tf.add(tf.matmul(X, weights), biases)

# 7. Declare the loss functions:
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

# 8. Initialize and train the model:
optimizer = tf.optimizers.SGD(learning_rate)
with tf.GradientTape() as tape:
    logits = tf.add(tf.matmul(X, weights), biases)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels, logits))
    gradients = tape.gradient(loss, [weights, biases])
    optimizer.apply_gradients(zip(gradients, [weights, biases]))

# 9. Evaluate the model:
print(f"final loss is: {loss.numpy():.3f}")
preds = tf.math.argmax(tf.add(tf.matmul(X, weights), biases),
                       axis=1)

# 10. Tune hyperparameters:
ground_truth = tf.math.argmax(labels, axis=1)
for y_true, y_pred in zip(ground_truth.numpy(), preds.numpy()):
    print(f"real label: {y_true} fitted: {y_pred}")

# 11. Deploy/predict new outcomes:

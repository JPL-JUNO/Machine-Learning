"""
@Title: working with matrices
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 14:01:24
@Description: 
"""
import tensorflow as tf
import numpy as np

# 1. Creating matrices:
identity_matrix = tf.linalg.diag([1.0, 1.0, 1.0])
A = tf.random.truncated_normal([2, 3])
B = tf.fill([2, 3], 5.0)
C = tf.random.uniform([3, 2])
D = tf.convert_to_tensor(np.array([[1., 2., 3.],
                                  [-3., -7., -1.],
                                  [0., 5., -2.]]),
                         dtype=tf.float32)
# 2. Addition, subtraction, and multiplication:
print(A + B)
print(B - B)
print(tf.matmul(B, identity_matrix))
# It is important to note that the matmul() function has arguments that specify whether
# or not to transpose the arguments before multiplication (the Boolean parameters,
# transpose_a and transpose_b), or whether each matrix is sparse (a_is_sparse and
# b_is_sparse).

print(tf.multiply(D, identity_matrix))
# Note that matrix division is not explicitly defined. While many
# define matrix division as multiplying by the inverse, it is
# fundamentally different from real-numbered division.

# 3. The transpose:
print(tf.transpose(C))

# 4. Determinant:
print(tf.linalg.det(D))

# 5. Inverse: To find the inverse of a square matrix
print(tf.linalg.inv(D))

# 6. Decompositions:
print(tf.linalg.cholesky(identity_matrix))

# 7. Eigenvalues and eigenvectors:
print(tf.linalg.eigh(D))
# Note that the tf.linalg.eigh() function outputs two tensors:
# in the first, you find the eigenvalues and,
# in the second tensor, you have the eigenvectors.

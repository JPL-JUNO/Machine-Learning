"""
@Description: Functions that are used to provide convenience
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-14 20:47:56
"""
from scipy.special import comb
import math
import re
from numpy import ndarray
import numpy as np


def ensemble_error(n_classifier: int, error: float):
    assert error <= 1
    assert error >= 0
    k_start = int(math.ceil(n_classifier / 2))
    probs = [comb(n_classifier, k) * error**k * (1 - error) **
             (n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probs)


def preprocessor(text: str) -> list:
    text = re.sub('<[^>]*>', '', text)  # remove all of th HTML markup
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emotions).replace('-', ''))
    return text


def tokenizer(text: str) -> list:
    return text.split()


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()


def tokenizer_porter(text: str) -> list:
    return [porter.stem(word) for word in text.split()]


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')


def tokenizer2(text: str) -> list:
    text = preprocessor(text)
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def mean_absolute_deviation(data: ndarray) -> float:
    """mean absolute deviation (median absolute deviation, MAD)

    Args:
        data (ndarray): _description_

    Returns:
        float: _description_
    """
    assert len(data.shape) == 1
    return np.mean(np.abs(data - np.mean(data)))


def sigmoid(z: ndarray) -> ndarray:
    """
    _summary_

    Parameters
    ----------
    z : ndarray
        _description_

    Returns
    -------
    ndarray
        _description_
    """
    return 1 / (1 + np.exp(-z))


def int_to_onehot(y: ndarray, num_labels: int) -> ndarray:
    """
    将一个预测向量转化为ndarray形式的独热编码

    Parameters
    ----------
    y : ndarray
        预测向量
    num_labels : int
        不同类型标签的数量

    Returns
    -------
    ndarray
        独热形式表示的预测向量
    Examples:
    >>> int_to_onehot(np.array([1, 3, 5, 6]), 7)
    >>> array([[0., 1., 0., 0., 0., 0., 0.],
    ...        [0., 0., 0., 1., 0., 0., 0.],
    ...        [0., 0., 0., 0., 0., 1., 0.],
    ...        [0., 0., 0., 0., 0., 0., 1.]])

    """
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


def minibatch_generator(X: ndarray, y: ndarray, minibatch_size: int):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]


def mse_loss(targets, probas, num_labels: int = 10):
    onehot_targets = int_to_onehot(
        targets, num_labels=num_labels
    )
    return np.mean((onehot_targets - probas)**2)


def accuracy(targets, predicted_labels) -> float:
    return np.mean(predicted_labels == targets)


def compute_mse_and_acc(nnet, X, y,
                        num_labels: int = 10,
                        minibatch_size: int = 100) -> tuple[float, float]:
    mse, correct_pred, num_examples = 0.0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probs = nnet.forward(features)
        predicted_labels = np.argmax(probs, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probs) ** 2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse / i
    acc = correct_pred / num_examples
    return mse, acc


if __name__ == '__main__':
    pass

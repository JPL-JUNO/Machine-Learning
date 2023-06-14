"""
@Description: Functions that are used to provide convenience
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-14 20:47:56
"""
from scipy.special import comb
import math


def ensemble_error(n_classifier: int, error: float):
    assert error <= 1
    assert error >= 0
    k_start = int(math.ceil(n_classifier / 2))
    probs = [comb(n_classifier, k) * error**k * (1 - error) **
             (n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probs)

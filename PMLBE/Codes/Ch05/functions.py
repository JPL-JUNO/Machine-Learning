"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 22:55:47
"""

import numpy as np
from numpy import ndarray


def sigmoid(input: ndarray) -> ndarray:
    return 1.0 / (1 + np.exp(-input))


if __name__ == '__main__':
    pass

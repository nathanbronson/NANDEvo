from typing import Iterable

import numpy as np

def acc(y: Iterable[bool], y_pred: Iterable[bool]) -> float:
    """
    accuracy of binary y_pred to y
    Arguments:
        y: Iterable[bool] = target values
        y_pred: Iterable[bool] = predicted values
    Returns:
        float
    """
    return 1 - np.mean(np.abs(y ^ y_pred)).astype(np.float32)
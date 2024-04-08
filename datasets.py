from typing import Iterable, Any

import numpy as np
from numpy.typing import NDArray

from functools import partial
from scheme import Scheme, define_scheme

_br = np.vectorize(np.binary_repr)
_l = np.vectorize(len)

def str_pad(pl: int, s: str) -> str:
    """
    string s padded to length pl with "0"
    Arguments:
        pl: int = length to be padded to
        s: str = string to be padded
    Returns:
        str
    """
    return ("0" * pl)[:-len(s)] + s

def br(a: Iterable[int]) -> Iterable[int]:
    """
    binary representation of all elements in a padded to the same length
    Arguments:
        a: Iterable[int] = array to be padded
    Returns:
        [int]
    """
    a = _br(a)
    pad_len = int(np.max(np.log10(a.astype(np.int32))) + 1)
    _sp = np.vectorize(partial(str_pad, pad_len))
    return _sp(a)

def l_merge(a: Iterable[Iterable[Any]]) -> Iterable[Any]:
    """
    first two elements of a list a merged to one list
    Arguments:
        a: [[Any]] = list of lists
    Returns:
        [Any]
    """
    return list(a[0]) + list(a[1])

def l_split(a: Iterable[str]) -> Iterable[str]:
    """
    first element in a split into a list of characters
    Arguments:
        a: Iterable[str] = list with string to be split
    Returns:
        [str]
    """
    return list(a[0])

class Addition(object):
    """
    Integer Addition Dataset holds information for training on addition operation
    """
    def __init__(self, min_num: int, max_num: int) -> None:
        """
        initialize Integer Addition Dataset with addition of all values from min_num to max_num
        Arguments:
            min_num: int = minimum number to be included as an addend
            max_num: int = maximum number to be incuded as an addend
        Returns:
            None
        """
        r = np.arange(max_num - min_num + 1) + min_num
        a, b = np.meshgrid(r, r)
        x = np.concatenate([np.reshape(a, (-1, 1)), np.reshape(b, (-1, 1))], axis=-1)
        y = np.sum(x, axis=-1)
        x = br(x)
        y = br(y)
        self.scheme: Scheme = define_scheme(2 * np.max(_l(x)), np.max(_l(y)), extra_streams=10)
        x = np.apply_along_axis(l_merge, 1, x)
        y = np.apply_along_axis(l_split, 1, np.expand_dims(y, 1))
        self.x: NDArray[bool] = x.astype(bool)
        self.y: NDArray[bool] = y.astype(bool)
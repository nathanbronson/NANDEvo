from typing import Iterable, Any, Union

import numpy as np

def unique_no_sort(a: Iterable[Any]) -> Iterable[Any]:
    """
    unique values of in the order they appear in a
    Arguments:
        a: [Any]
    Returns:
        [Any]
    """
    a, idx = np.unique(a, return_index=True)
    idx = crunch(idx)
    return a[idx]

def crunch(a: Iterable[int]) -> Iterable[int]:
    """
    assign sequential integers in the range of the length of a to elements of a by order
    Arguments:
        a: [int] = array to be crunched
    Returns:
        [int]
    """
    r = np.arange(a.shape[0])
    for i in r:
        while np.sum(a == i) < 1:
            a[a > i] -= 1
    return a

def off_pad(pad_len: Union[int, float], x: Iterable[str]) -> Iterable[str]:
    """
    x padded with "" to be pad_len
    Arguments:
        pad_len: int, float = length to which x should be padded
        x: [str] = sequence to pad
    Returns:
        [str]
    """
    return np.concatenate([([""] * int(pad_len))[:-len(x)], x])
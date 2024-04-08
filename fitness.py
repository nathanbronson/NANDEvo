from functools import partial
from typing import Iterable, Callable, Dict
from types import FunctionType as function

import numpy as np

from function_translation import instructions_to_function
from scheme import Scheme

def fitness_from_func(x: Iterable[Iterable[bool]], y: Iterable[bool], loss: Callable[[Iterable[bool], Iterable[bool]], float], func: Iterable[Callable[[Iterable[bool]], Iterable[bool]]]) -> float:
    """
    fitness of given func based on accuracy of predictions of y from x computed by loss
    Arguments:
        x: [[bool]] = sample inputs
        y: [bool] = expected outputs
        loss: ([bool], [bool]) -> float = loss from expected and computed y values
        func: [([bool]) -> [bool]] = function to be evaluated
    Returns:
        float
    """
    if type(func) != function:
        func = func[0]
    y_pred = np.apply_along_axis(func, 1, x)
    return loss(np.squeeze(y), np.squeeze(y_pred))

def fitness_from_instructions(x: Iterable[Iterable[bool]], y: Iterable[bool], scheme: Scheme, loss: Callable[[Iterable[bool], Iterable[bool]], float], instrs: Iterable[str]) -> Dict[str, Iterable[float]]:
    """
    fitness of given instructions based on accuracy of predictions of y from x computed by loss
    Arguments:
        x: [[bool]] = sample inputs
        y: [bool] = expected outputs
        scheme: Scheme = scheme for instruction translation
        loss: ([bool], [bool]) -> float = loss from expected and computed y values
        instrs: [str] = list of instructions to be converted to functions and evaluated
    Returns:
        float
    """
    vec_itf = np.vectorize(partial(instructions_to_function, scheme))
    funcs = vec_itf(instrs)
    return {i: f for i, f in zip(instrs, np.apply_along_axis(partial(fitness_from_func, x, y, loss), 1, np.expand_dims(funcs, 1)).tolist())}
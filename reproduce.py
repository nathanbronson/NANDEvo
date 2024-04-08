from random import randint
from functools import partial
from typing import Tuple, Iterable, Dict

import numpy as np

from utils import off_pad

TARGET_POP_SIZE = 100
TARGET_POP_SIZE /= 3
TARGET_POP_SIZE = int(TARGET_POP_SIZE)

def crossover(p1: str, p2: str) -> Tuple[str, str]:
    """
    crossover of p1 and p2
    Arguments:
        p1: str = instructions of parent 1
        p2: str = instructions of parent 2
    Returns:
        (str, str)
    """
    if len(p1) <= 1 or len(p2) <= 1:
        return p1, p2
    p1_idx = randint(0, len(p1) - 1)
    p2_idx = randint(0, len(p2) - 1)
    return p1[:p1_idx] + p2[p2_idx:], p2[:p2_idx] + p1[p1_idx:]
l_cross = lambda x: crossover(x[0], x[1])

def _mut(choices: str, mut_prob: float, _p: str) -> str:
    """
    single element _p mutated
    Arguments:
        choices: str = all possible mutation results
        mut_prob: float = probability of mutation occurence on [0, 1]
        _p: str = non-mutated element
    Returns:
        str
    """
    return np.random.choice([_p, np.random.choice(list(choices))], p=[1 - mut_prob, mut_prob])

def mutate(choices: str, p: str, mut_prob: float = .3) -> str:
    """
    instructions p mutated
    Arguments:
        choices: str = all possible mutation results
        p: str = non-mutated instructions
        mut_prob: float = probability of mutation occurence on [0, 1]
    Returns:
        str
    """
    return "".join(np.vectorize(partial(_mut, choices + " ", mut_prob))(np.array(list(" " + p + " "))).tolist()).replace(" ", "")

def reproduce(fitnesses: Dict[str, float], choices: str, parent: str, target_size: int = TARGET_POP_SIZE) -> Iterable[str]:
    """
    result of reproduction by parent in population represented by fitnesses
    Arguments:
        fitnesses: {str: float} = instructions in population and the fitness of each
        choices: str = all possible mutation results
        parent: str = parent instructions reproducing
        target_size: int = target population size
    Returns:
        [str]
    """
    vec_mut = np.vectorize(partial(mutate, choices))
    f_prob = np.array(list(fitnesses.values()))
    s_f_prob = np.sum(f_prob)
    n_children = int((fitnesses[parent] / s_f_prob) * target_size)
    if n_children <= 0:
        return [""]
    f_prob /= s_f_prob
    return np.reshape(vec_mut(np.apply_along_axis(l_cross, 1, np.concatenate([np.repeat([[parent]], n_children, axis=0), np.expand_dims(np.random.choice(list(fitnesses.keys()), size=n_children, p=f_prob, replace=True), -1)], axis=-1))), (-1,))

def select(population: Iterable[str], target_size: int = TARGET_POP_SIZE) -> Iterable[str]:
    """
    selected population (only unique non-null members)
    Arguments:
        population: [str] = full population
        target_size: int = target population size
    Returns:
        [str]
    """
    return np.unique(population[population != ""])

def fit_select(fitnesses: Dict[str, float], target_size: int = TARGET_POP_SIZE) -> Dict[str, float]:
    """
    top target_size individuals from fitnesses
    Arguments:
        fitnesses: {str: float} = individuals with fitnesses
        target_size: int = target population size
    Returns:
        {str: float}
    """
    if len(fitnesses) <= target_size:
        return fitnesses
    return {k: v for k, v in fitnesses.items() if v >= sorted(fitnesses.values(), reverse=True)[target_size - 1]}

def genesis(fitnesses: Dict[str, float], choices: str, target_size: int = TARGET_POP_SIZE) -> Iterable[str]:
    """
    full population in fitnesses reproduced
    Arguments:
        fitnesses: {str: float} = instructions in population and the fitness of each
        choices: str = all possible mutation results
        target_size: int = target population size
    Returns:
        [str]
    """
    instrs = np.expand_dims(np.array([i[0] for i in sorted(list(fitnesses.items()), reverse=True, key=lambda x: x[1])[:int(target_size)]]), -1)
    offspring = np.apply_along_axis(lambda x: off_pad(target_size * 2.5, reproduce(fitnesses, choices, x[0], target_size=2*target_size)), 1, instrs)
    return select(np.concatenate([np.reshape(offspring, (-1,)), np.reshape(instrs, (-1,))]), target_size)
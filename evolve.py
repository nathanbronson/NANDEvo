from typing import Iterable, Callable, List, Optional, Tuple

import numpy as np

from fitness import fitness_from_instructions
from reproduce import genesis, fit_select
from losses import acc
from scheme import Scheme
from utils import unique_no_sort

def evolve(x: Iterable[Iterable[bool]], y: Iterable[bool], scheme: Scheme, loss: Callable[[Iterable[bool], Iterable[bool]], float] = acc, initial_population: List[str] = [], num_generations: Optional[int] = None, target_fitness: float = 1e5, logging: int = 5, top_seed: int = 20) -> Tuple[float, str]:
    """
    evolved function to approximate y from x
    Arguments:
        x: [[bool]] = x values from which y is approximated
        y: [bool] = y values to approximate from x
        scheme: Scheme = scheme to be used
        loss: ([bool], [bool]) -> float = loss function to evaluate fitness
        initial_population: [str] = inital population for evolution
        num_generations: int = number of generations for which the program should run
        target_fitness: float = target fitness at which evolution terminates
        logging: int = logging interval
        top_seed: int = number of generation bests to seed in population
    Returns:
        (float, str)
    """
    population = initial_population if len(initial_population) > 0 else list(scheme.instruction_set[:-1])
    generation_bests = []
    generation = 0
    while True:
        generation += 1
        try:
            fitnesses = fitness_from_instructions(x, y, scheme, loss, population)
            fitnesses = fit_select(fitnesses)
            generation_bests.append(max([i[::-1] for i in fitnesses.items()]))
            if generation_bests[-1][0] >= target_fitness:
                break
            population = np.concatenate([genesis(fitnesses, scheme.instruction_set), unique_no_sort([i[1] for i in sorted(generation_bests, reverse=True)])[:top_seed]])
        except KeyboardInterrupt:
            break
        if generation % logging == 0:
            print("Generation {0} -- npop: {4} - max: {1:.6} - gavg: {3:.6} - alen {5:.3}".format(str(generation), str(max(generation_bests)[0]), str(generation_bests[-1][0]), str(sum(list(fitnesses.values()))/len(fitnesses)), len(population), np.mean([len(i) for i in fitnesses.keys()]).tolist()), end="\r")
        if num_generations is not None:
            if generation >= num_generations:
                break
    print("Final Generation: {0} - npop: {4} - max: {1:.6} - - gavg: {3:.6} - alen {5:.3}".format(str(generation), str(max(generation_bests)[0]), str(generation_bests[-1][0]), str(sum(list(fitnesses.values()))/len(fitnesses)), len(population), np.mean([len(i) for i in fitnesses.keys()]).tolist()))
    best = max(generation_bests)
    print("Fittest Function:")
    print(best[1])
    print("fitness: {}".format(best[0]))
    return best
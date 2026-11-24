'''
Simulated annealing: search for the charge configuration x* = arg min J(x), x in the feasible set.

Owner: feature/simulated-annealing. The algorithm is generic: it receives the objective and a
neighbour function, so it does not depend on what x contains (coil grouping, order, T_sat, t_sat).
'''

from collections.abc import Callable

import numpy as np


def optimize(*, objective: Callable[[dict], float], x0: dict, neighbor: Callable[[dict, np.random.Generator], dict], T0: float = 1.0, alpha: float = 0.95, n_iter: int = 1000, seed: int = 42) -> dict:
    '''
    Minimize an objective with simulated annealing and a geometric cooling schedule.

    Args:
        objective: function x -> J(x), lower is better (objective_function.objective)
        x0:        initial candidate charge
        neighbor:  function (x, rng) -> new candidate close to x
        T0:        initial temperature of the algorithm, in the units of J
        alpha:     cooling factor per iteration, in (0, 1)
        n_iter:    number of iterations
        seed:      random seed

    Returns:
        Dict with x_best, J_best and history (list of dicts with iteration, T, J, J_best, accepted).

    Raises:
        NotImplementedError until the simulated-annealing branch implements it.
    '''
    raise NotImplementedError('feature/simulated-annealing')


def neighbor_swap_coils(x: dict, rng: np.random.Generator) -> dict:
    '''
    Example neighbour: swap one coil of the charge with one from the pool of available coils.

    Args:
        x:      candidate charge dict, with the extra key pool (coils not in the charge)
        rng:    numpy random generator

    Returns:
        New candidate charge.

    Raises:
        NotImplementedError until the simulated-annealing branch implements it.
    '''
    raise NotImplementedError('feature/simulated-annealing')

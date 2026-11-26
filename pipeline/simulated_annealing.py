'''Model-independent minimization by simulated annealing.'''

from collections.abc import Callable, Iterable
from copy import deepcopy
import math
from numbers import Integral, Real
from typing import TypeVar

import numpy as np

State = TypeVar('State')


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _evaluate(objective, feasible, state):
    '''Call each callback on a private copy. Propagate errors from the model.'''
    if feasible is not None:
        valid = feasible(deepcopy(state))
        if not isinstance(valid, (bool, np.bool_)):
            raise TypeError('feasible must return a boolean')
        if not valid:
            return None, 'infeasible'
    score = objective(deepcopy(state))
    if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real):
        raise TypeError('objective must return a real scalar')
    score = float(score)
    return (score, 'finite') if math.isfinite(score) else (None, 'nonfinite')


def estimate_initial_temperature(deltas: Iterable[float], target_acceptance: float = 0.8) -> float:
    '''Set P(accept) to the target for the median positive finite pilot increment.

    The target describes this representative uphill step, not every proposal.
    Supply differences J(neighbor) - J(anchor) on the objective scale of interest.
    '''
    p = _positive('target_acceptance', target_acceptance)
    if p >= 1:
        raise ValueError('target_acceptance must be in (0, 1)')
    uphill = sorted(d for d in map(float, deltas) if math.isfinite(d) and d > 0)
    if not uphill:
        raise ValueError('pilot has no finite positive increments')
    # Midpoint without overflowing the sum of the two central values.
    left, right = uphill[(len(uphill) - 1) // 2], uphill[len(uphill) // 2]
    typical = left + (right - left) / 2
    return _positive('estimated T0', typical / -math.log(p))


def cooling_factor(T0: float, T_final: float, n_iter: int) -> float:
    '''Cool from T0 on proposal 1 to T_final on proposal n_iter (N-1 intervals).'''
    start, stop = _positive('T0', T0), _positive('T_final', T_final)
    n = _integer('n_iter', n_iter, 2)
    if stop >= start:
        raise ValueError('T_final must be smaller than T0')
    alpha = math.exp((math.log(stop) - math.log(start)) / (n - 1))
    if not 0 < alpha < 1:
        raise ValueError('this schedule has no representable cooling factor in (0, 1)')
    return alpha


def _accept(current: float, proposed: float, temperature: float, rng: np.random.Generator) -> bool:
    if proposed <= current:
        return True
    if temperature == 0:  # Geometric cooling below the range of floating-point numbers.
        return False
    delta = proposed - current
    scaled = (delta / temperature if math.isfinite(delta)
              else proposed / temperature - current / temperature)
    return rng.random() < math.exp(-scaled)


def optimize(*, objective: Callable[[State], float], x0: State,
             neighbor: Callable[[State, np.random.Generator], State],
             T0: float = 1.0, alpha: float = 0.95, n_iter: int = 1000, seed: int = 42,
             feasible: Callable[[State], bool] | None = None) -> dict:
    '''Minimize J with one proposal per iteration, starting at a feasible finite x0.

    The optional feasible predicate runs before the objective. Rejected proposals
    still consume one iteration and cool the algorithm. Callback exceptions propagate.
    The neighbor may mutate its argument and must use the supplied RNG for replay.
    States must support deepcopy; the objective should be deterministic.

    T in row k (1-based) is the algorithmic temperature T0 * alpha**(k-1),
    in objective units. J and J_best are recorded after the decision. The initial
    evaluation is counted in n_evaluations but is not a proposal or a history row.
    '''
    if not callable(objective) or not callable(neighbor) or (feasible is not None and not callable(feasible)):
        raise TypeError('objective, neighbor and any feasible predicate must be callable')
    T0 = _positive('T0', T0)
    alpha = _positive('alpha', alpha)
    if alpha >= 1:
        raise ValueError('alpha must be in (0, 1)')
    n_iter = _integer('n_iter', n_iter, 1)
    rng = np.random.default_rng(_integer('seed', seed, 0))

    current = deepcopy(x0)
    current_value, status = _evaluate(objective, feasible, current)
    if status != 'finite':
        raise ValueError(f'x0 is {status}')
    initial_value = current_value
    best, best_value = deepcopy(current), current_value
    history = []
    evaluations, accepted_count = 1, 0
    log_start, log_alpha = math.log(T0), math.log(alpha)
    for k in range(n_iter):
        temperature = T0 if k == 0 else math.exp(log_start + k * log_alpha)
        proposal = neighbor(deepcopy(current), rng)
        value, status = _evaluate(objective, feasible, proposal)
        evaluations += int(status != 'infeasible')
        accepted = value is not None and _accept(current_value, value, temperature, rng)
        if accepted:
            current, current_value = deepcopy(proposal), value
            accepted_count += 1
            if value < best_value:
                best, best_value = deepcopy(current), value
        history.append({'iteration': k + 1, 'T': temperature, 'J': current_value,
                        'J_best': best_value, 'accepted': accepted, 'status': status,
                        'acceptance_rate': accepted_count / (k + 1)})
    return {'x_best': best, 'J_best': best_value, 'history': history,
            'x_current': current, 'J_current': current_value, 'J_initial': initial_value,
            'n_evaluations': evaluations,
            'acceptance_rate': accepted_count / n_iter}


def neighbor_swap_coils(x: dict, rng: np.random.Generator) -> dict:
    '''Exchange whole records between a charge and its pool, preserving coil properties.'''
    candidate = deepcopy(x)
    coils, pool = candidate['coils'], candidate['pool']
    if not coils or not pool:
        raise ValueError('exchange requires a nonempty charge and pool')
    i, j = rng.integers(len(coils)), rng.integers(len(pool))
    coils[i], pool[j] = pool[j], coils[i]
    return candidate

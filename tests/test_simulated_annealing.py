'''Behavioral checks of the generic optimizer on synthetic objectives only.'''

from copy import deepcopy
import math

import numpy as np
import pytest

from pipeline.simulated_annealing import (cooling_factor, estimate_initial_temperature,
                                          neighbor_swap_coils, optimize)
from scripts.verify_simulated_annealing import METHODS, problems, run_trial


def test_metropolis_threshold():
    # For a +1 move at log(2)^(-1), p=0.5. Both decisions are reproducible.
    for seed in (2, 0):
        u = np.random.default_rng(seed).random()
        result = optimize(x0=1, objective=float, neighbor=lambda x, rng: x + 1,
                          T0=1 / math.log(2), n_iter=1, seed=seed)
        assert result['history'][0]['accepted'] == (u < 0.5)
        assert result['J_current'] == (2 if u < 0.5 else 1)
        assert result['J_best'] == 1


def test_in_place_moves_and_best_survive_rejection_and_uphill_acceptance():
    start = {'nested': [5]}
    moves = iter([1, 2, 10000, 2])
    def neighbor(x, rng):
        x['nested'][0] = next(moves)
        return x

    def objective(x):
        return x['nested'].pop()

    r = optimize(x0=start, objective=objective, neighbor=neighbor,
                 n_iter=4, T0=100, alpha=0.8, seed=2)
    assert [row['accepted'] for row in r['history']] == [True, True, False, True]
    assert [row['J'] for row in r['history']] == [1, 2, 2, 2]
    assert [row['acceptance_rate'] for row in r['history']] == [1, 1, 2/3, 0.75]
    assert r['acceptance_rate'] == 0.75
    assert r['x_current'] == {'nested': [2]}
    r['x_current']['nested'].clear()
    assert r['x_best'] == {'nested': [1]} and start == {'nested': [5]}


def test_invalid_proposals_and_actual_evaluation_count():
    proposals = iter([-1, 2, 3, 4, 0])
    seen = []
    def objective(x):
        seen.append(x)
        return {2: np.nan, 3: np.inf, 4: -np.inf}.get(x, x)

    r = optimize(x0=1, objective=objective, neighbor=lambda x, rng: next(proposals),
                 feasible=lambda x: x >= 0, n_iter=5, T0=4, alpha=0.5)
    assert seen == [1, 2, 3, 4, 0] and r['n_evaluations'] == len(seen)
    assert [h['status'] for h in r['history']] == ['infeasible', 'nonfinite', 'nonfinite', 'nonfinite', 'finite']
    assert [h['J'] for h in r['history']] == [1, 1, 1, 1, 0]
    assert [h['T'] for h in r['history']] == pytest.approx([4, 2, 1, 0.5, 0.25])


@pytest.mark.parametrize('overrides', [
    {'T0': 0}, {'T0': np.nan}, {'alpha': 1}, {'alpha': -1}, {'alpha': np.inf},
    {'n_iter': 0}, {'n_iter': 2.5}, {'n_iter': True}, {'seed': -1}, {'seed': 2.5},
    {'feasible': lambda x: False}, {'objective': lambda x: np.nan},
])
def test_invalid_arguments_and_initial_state(overrides):
    args = dict(x0=0, objective=float, neighbor=lambda x, rng: x)
    with pytest.raises(ValueError):
        optimize(**(args | overrides))


def test_callback_errors_propagate():
    def broken(x, rng):
        raise KeyError('missing field')
    with pytest.raises(KeyError, match='missing field'):
        optimize(x0=1, objective=float, neighbor=broken)


def test_conventional_known_problems_and_seed_replay():
    for problem, tolerance in zip(problems(), (0.02, 0.5, 15)):
        args = dict(x0=problem.start, objective=problem.objective, neighbor=problem.neighbor,
                    seed=3, n_iter=2000)
        a, b = optimize(**args), optimize(**args)
        assert a['history'] == b['history']
        assert np.array_equal(a['x_best'], b['x_best'])
        assert problem.optimum <= a['J_best'] <= problem.optimum + tolerance
        assert a['J_best'] == problem.objective(a['x_best'])
        assert a['J_best'] < problem.objective(problem.start)


def test_swap_preserves_records_and_input():
    x = {'coils': [{'id': 1, 'mass': 2}], 'pool': [{'id': 2, 'mass': 5}], 'T_sat': 900}
    original = deepcopy(x)
    y = neighbor_swap_coils(x, np.random.default_rng(0))
    assert y == {'coils': x['pool'], 'pool': x['coils'], 'T_sat': 900}
    y['coils'][0]['mass'] = 0
    assert x == original


def test_calibration_and_last_used_temperature():
    deltas = [-4, 0, 1, 3, 7, 100, np.nan, np.inf]
    T0 = estimate_initial_temperature(deltas, 0.75)
    assert math.exp(-5 / T0) == pytest.approx(0.75)
    alpha = cooling_factor(T0, 0.002, 11)
    r = optimize(x0=1, objective=float, neighbor=lambda x, rng: x,
                 T0=T0, alpha=alpha, n_iter=11)
    assert r['history'][0]['T'] == T0
    assert r['history'][-1]['T'] == pytest.approx(0.002)


@pytest.mark.parametrize('deltas,p', [([], 0.8), ([-1, 0, np.inf], 0.8), ([1], 0), ([1], 1)])
def test_invalid_calibration(deltas, p):
    with pytest.raises(ValueError):
        estimate_initial_temperature(deltas, p)


@pytest.mark.parametrize('parameters', [(1, 0, 20), (2, 2, 10), (1, 0.2, 1)])
def test_invalid_endpoints(parameters):
    with pytest.raises(ValueError):
        cooling_factor(*parameters)


def test_budget_includes_pilot_and_all_methods_replay():
    problem = problems()[0]
    calls = []
    original = problem.objective
    def counted(x):
        calls.append(1)
        return original(x)

    problem.objective = counted
    for method in METHODS:
        runs = []
        for _ in range(2):
            calls.clear()
            row, traces = run_trial(problem, method, seed=5, budget=121)
            assert len(calls) == row['evaluations'] == 121
            assert row['pilot_evaluations'] == (64 if method == 'Calibrado' else 0)
            runs.append((row, traces))
        assert runs[0][0] == runs[1][0]
        for a, b in zip(runs[0][1], runs[1][1]):
            np.testing.assert_array_equal(a, b)

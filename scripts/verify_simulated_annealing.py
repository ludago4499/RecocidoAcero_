'''Reproduce the synthetic comparison with python -m scripts.verify_simulated_annealing.'''

import argparse
import csv
from dataclasses import dataclass
from itertools import permutations
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pipeline.simulated_annealing import cooling_factor, estimate_initial_temperature, optimize

OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'sa_verification'
METHODS = ('Convencional', 'Calibrado', 'Aleatorio')

@dataclass
class Problem:
    name: str
    start: np.ndarray
    objective: Callable
    neighbor: Callable
    sample: Callable
    optimum: float


def sphere(x):
    return float(x @ x)


def rastrigin(x):
    return float(x @ x + 10 * np.sum(1 - np.cos(2 * np.pi * x)))


BOUND = 5.12


def box_neighbor(x, rng):
    '''Reflect a Gaussian step at the faces of the square [-BOUND, BOUND]^2.'''
    trial = x + rng.normal(0, 0.55, size=x.shape)
    return BOUND - np.abs((trial + BOUND) % (4 * BOUND) - 2 * BOUND)


def box_sample(rng):
    return rng.uniform(-BOUND, BOUND, size=2)


# Synthetic directed travel costs. Node 0 is the fixed depot.
COST = np.array([
    [0, 17, 8, 21, 12, 16, 13, 6], [9, 0, 14, 5, 18, 8, 7, 20],
    [16, 6, 0, 18, 9, 15, 11, 13], [14, 12, 5, 0, 17, 6, 19, 8],
    [10, 16, 13, 7, 0, 14, 6, 11], [8, 21, 12, 9, 15, 0, 10, 5],
    [19, 7, 16, 14, 8, 9, 0, 12], [13, 15, 7, 10, 6, 18, 11, 0],
])


def route_cost(order):
    '''Sum all directed legs, including departure from and return to the depot.'''
    return int(COST[0, order[0]] + sum(COST[a, b] for a, b in zip(order, order[1:]))
               + COST[order[-1], 0])


def swap_positions(order, rng):
    i, j = rng.choice(len(order), size=2, replace=False)
    candidate = order.copy()
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def problems():
    '''The route reference is obtained independently of the optimizer by enumeration.'''
    start = np.array([4.2, -3.8])
    cities = np.arange(1, len(COST))
    exact = min(route_cost(route) for route in permutations(cities))
    return [Problem('Sphere', start, sphere, box_neighbor, box_sample, 0.0),
            Problem('Rastrigin', start, rastrigin, box_neighbor, box_sample, 0.0),
            Problem('Ruta', cities, route_cost, swap_positions,
                    lambda rng: rng.permutation(cities), exact)]


def run_trial(problem, method, seed, budget=2001, pilot_pairs=32):
    '''Run one method with a measured, inclusive budget of objective evaluations.'''
    calls = 0
    def evaluate(x):
        nonlocal calls
        calls += 1
        return problem.objective(x)

    pilot_calls = 0
    T0, alpha, acceptance = None, None, None
    temp, accepted = np.array([]), np.array([])
    if method == 'Aleatorio':
        rng = np.random.default_rng([seed, 2])
        values = [evaluate(problem.start)]
        values.extend(evaluate(problem.sample(rng)) for _ in range(budget - 1))
        best = np.minimum.accumulate(values)
    else:
        if method == 'Calibrado':
            rng = np.random.default_rng([seed, 1])
            deltas = []
            for _ in range(pilot_pairs):
                anchor = problem.sample(rng)
                base = evaluate(anchor)
                deltas.append(evaluate(problem.neighbor(anchor, rng)) - base)
            pilot_calls = calls
            T0 = estimate_initial_temperature(deltas)
            alpha = cooling_factor(T0, T0 * 0.001, budget - pilot_calls - 1)
        elif method == 'Convencional':
            T0, alpha = 1.0, 0.95
        else:
            raise ValueError(f'unknown method: {method}')
        r = optimize(objective=evaluate, x0=problem.start, neighbor=problem.neighbor,
                     T0=T0, alpha=alpha, n_iter=budget - pilot_calls - 1, seed=seed)
        history = r['history']
        best = np.array([r['J_initial']] + [row['J_best'] for row in history])
        temp = np.array([row['T'] for row in history])
        accepted = np.array([row['accepted'] for row in history])
        acceptance = r['acceptance_rate']
    if calls != budget:
        raise RuntimeError(f'{method} used {calls} evaluations; expected {budget}')
    row = dict(problem=problem.name, method=method, seed=seed, evaluations=calls,
               pilot_evaluations=pilot_calls, optimum=problem.optimum, best=float(best[-1]),
               gap=float(best[-1] - problem.optimum), T0=T0, alpha=alpha,
               acceptance_rate=acceptance)
    return row, (best, temp, accepted)


def plot_problem(problem, traces, output):
    '''Median best cost, algorithmic temperature and batch acceptance across seeds.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout='constrained')
    convergence, temperature, acceptance = axes
    colors = ('#15598a', '#b04418', '#5e6674')
    for method, color in zip(METHODS, colors):
        rows = traces[method]
        best = np.stack([curves[0] for _, curves in rows])
        offset = rows[0][0]['pilot_evaluations']
        x = offset + np.arange(1, best.shape[1] + 1)
        q1, median, q3 = np.quantile(best, [0.25, 0.5, 0.75], axis=0)
        convergence.plot(x, median, color=color, label=method)
        convergence.fill_between(x, q1, q3, color=color, alpha=0.12, linewidth=0)
        if method != 'Aleatorio':
            temp = np.stack([curves[1] for _, curves in rows])
            temperature.plot(x[1:], np.median(temp, axis=0), color=color)
            flags = np.stack([curves[2] for _, curves in rows])
            batches = np.arange(0, flags.shape[1], 100)
            fractions = np.column_stack([flags[:, i:i+100].mean(axis=1) for i in batches])
            endpoints = np.minimum(batches + 100, flags.shape[1]) + offset + 1
            acceptance.plot(endpoints, np.median(fractions, axis=0), color=color, marker='.', ms=4)
    if problem.optimum == 0:
        convergence.set_yscale('log')
    else:
        convergence.axhline(problem.optimum, linestyle=':', color='#222222', label='Mínimo exacto')
    convergence.set(title='Convergencia', ylabel='Mejor costo encontrado')
    convergence.legend(frameon=False, fontsize=8)
    temperature.set(title='Enfriamiento', ylabel='Temperatura algorítmica', yscale='log')
    acceptance.set(title='Aceptación por bloque', ylabel='Propuestas aceptadas', ylim=(0, 1.03))
    acceptance.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel('Evaluaciones acumuladas')
        ax.grid(axis='y', alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{problem.name}  |  {len(rows)} semillas  ·  mínimo de referencia {problem.optimum:g}',
                 fontsize=15, weight='bold')
    fig.savefig(output / f'{problem.name.lower()}.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=2001, help='objective calls per seed and method')
    parser.add_argument('--seeds', type=int, default=20, help='number of seeds, starting at zero')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.budget < 67 or args.seeds < 1:
        parser.error('use at least 67 evaluations (64 pilot + initial + 2 proposals) and one seed')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for problem in problems():
        traces = {}
        for method in METHODS:
            runs = [run_trial(problem, method, seed, budget=args.budget) for seed in range(args.seeds)]
            rows = [row for row, _ in runs]
            results.extend(rows)
            traces[method] = runs
            scores = [row['best'] for row in rows]
            print(f'{problem.name:10s} {method:13s} min={min(scores):.6g} '
                  f'median={np.median(scores):.6g} max={max(scores):.6g} '
                  f'exact={scores.count(problem.optimum)}/{args.seeds}')
        plot_problem(problem, traces, args.output)
    with (args.output / 'runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()

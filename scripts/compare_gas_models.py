'''
Compare gas side models (channel model, bottom boundary, effective fan flow) on a sample of real charges.

For each configuration every sampled charge is simulated (heating only) and the model heating time is
regressed on load mass, mean strip thickness and number of coils, the same features as the plant fit
of t_net (+0.29 h/t, -0.3 h/mm, +1.0 h per coil). See docs/decision_log.md, D7 and D8.

Outputs (results/gas_models/, ignored by git because it derives from the confidential Excel)
    runs_<tag>.csv      one row per charge and configuration
    summary_<tag>.csv   one row per configuration: median heating time and fitted slopes

Usage, from the project root:
    python scripts/compare_gas_models.py --sample 60
'''

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bell_furnace.cases import build_load_case  # noqa: E402
from bell_furnace.furnace_parameters import COLD_SPOT_MARGIN, FURNACE_TERNIUM, HOUR  # noqa: E402
from bell_furnace.metrics import run_case, stack_metrics  # noqa: E402
from pipeline.data_processing import EXCEL_PATH, load_data  # noqa: E402

OUTPUT = ROOT / 'results' / 'gas_models'

CONFIGS = {
    'simple': {'channel_model': 'simple'},
    'scaled_yang_f1.00': {'channel_model': 'scaled', 'bottom_mode': 'yang', 'flow_scale': 1.0},
    'scaled_yang_f0.33': {'channel_model': 'scaled', 'bottom_mode': 'yang', 'flow_scale': 0.33},
    'scaled_base_f1.00': {'channel_model': 'scaled', 'bottom_mode': 'base_convector', 'flow_scale': 1.0},
    'scaled_base_f0.33': {'channel_model': 'scaled', 'bottom_mode': 'base_convector', 'flow_scale': 0.33},
    'scaled_base_f0.33_top0.15': {'channel_model': 'scaled', 'bottom_mode': 'base_convector', 'flow_scale': 0.33, 'top_gap': 0.15},
    'scaled_base_f0.33_top0.50': {'channel_model': 'scaled', 'bottom_mode': 'base_convector', 'flow_scale': 0.33, 'top_gap': 0.50},
    'scaled_base_f0.33_base0.05': {'channel_model': 'scaled', 'bottom_mode': 'base_convector', 'flow_scale': 0.33, 'base_gap': 0.05},
    'scaled_base_f0.33_base0.20': {'channel_model': 'scaled', 'bottom_mode': 'base_convector', 'flow_scale': 0.33, 'base_gap': 0.20},
}

_COILS = None
_LOADS = None


def init_worker(path: str) -> None:
    '''
    Read the Excel once in each worker process.

    Args:
        path:   path of Recocido -Tec.xlsx
    '''
    global _COILS, _LOADS
    _COILS, _LOADS = load_data(path=Path(path))


def simulate(task: tuple[int, str]) -> dict:
    '''
    Simulate one charge with one gas side configuration.

    Args:
        task:   tuple (load_id, configuration name in CONFIGS)

    Returns:
        Dict with load_id, config, features, plant times and model heating time in h.
    '''
    load_id, name = task
    load = _LOADS.loc[load_id]
    row = {'load_id': load_id, 'config': name, 'mass_t': load['load_mass'] / 1000.0, 'thickness_mm': load['thickness_mean'] * 1000.0, 'n_coils': load['n_coils'], 'base_id': load['base_id'], 't_net_h': load['t_net'] / HOUR}
    try:
        case = build_load_case(load_id=load_id, coils_table=_COILS, loads_table=_LOADS, n_r=11, n_z=15)
        case['furnace'] = {**FURNACE_TERNIUM, **CONFIGS[name]}
        results = run_case(case=case, heating_only=True, dt_max=120.0)
        m = stack_metrics(case=case, results=results, margin=COLD_SPOT_MARGIN)
        row.update({'t_heat_h': m['t_heat_cold'] / HOUR, 'slowest_coil': m['slowest_coil'], 'error': ''})
    except Exception as error:
        row.update({'t_heat_h': np.nan, 'error': f'{type(error).__name__}: {error}'})
    return row


def slopes(*, table: pd.DataFrame, target: str) -> dict:
    '''
    Fit target = a + b mass + c thickness + d n_coils by least squares.

    Args:
        table:  rows with columns mass_t, thickness_mm, n_coils and target
        target: column to fit

    Returns:
        Dict with per_t, per_mm, per_coil (h per unit) and r2.
    '''
    t = table.dropna(subset=[target])
    X = np.column_stack([np.ones(len(t)), t['mass_t'], t['thickness_mm'], t['n_coils']])
    y = t[target].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    return {'per_t': beta[1], 'per_mm': beta[2], 'per_coil': beta[3], 'r2': r2}


def main() -> None:
    '''
    Run every configuration on the sample and write runs.csv and summary.csv.
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sample', type=int, default=60, help='number of valid charges, 0 for all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--configs', nargs='*', default=list(CONFIGS)[:5])
    parser.add_argument('--tag', default='main', help='suffix of the output files')
    args = parser.parse_args()

    _, loads = load_data(path=EXCEL_PATH)
    ids = loads.index[loads['valid']].to_numpy()
    if args.sample:
        ids = np.random.default_rng(args.seed).choice(ids, size=min(args.sample, len(ids)), replace=False)
    tasks = [(int(i), c) for c in args.configs for i in ids]

    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(str(EXCEL_PATH),)) as pool:
        runs = pd.DataFrame(list(pool.map(simulate, tasks, chunksize=4)))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUTPUT / f'runs_{args.tag}.csv', index=False)
    first = runs[runs['config'] == args.configs[0]]
    rows = [{'config': 'plant t_net', 'median_h': first['t_net_h'].median(), **slopes(table=first, target='t_net_h')}]
    for name, group in runs.groupby('config', sort=False):
        rows.append({'config': name, 'median_h': group['t_heat_h'].median(), 'not_reached': int(group['t_heat_h'].isna().sum()), **slopes(table=group, target='t_heat_h')})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / f'summary_{args.tag}.csv', index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

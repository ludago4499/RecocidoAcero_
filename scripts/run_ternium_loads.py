'''
Simulate the real charges of Recocido -Tec.xlsx and write model and data side by side.

Each charge is built from its own coils (3 to 5, sizes and grades from the data, outer diameter from
the mass) and its own saturation temperature and time. Only the heating is simulated (see
metrics.heating_schedule). The output table is the input of the comparison, the PCA and the
predictive model.

Outputs (results/ternium_loads/)
    model_vs_data.csv        one row per charge: data features, data times and model metrics (h)
    model_vs_data.png        model heating time against the data, and against the charge mass

Usage, from the project root:
    python scripts/run_ternium_loads.py                        every valid charge, all CPU cores
    python scripts/run_ternium_loads.py --sample 40            40 random valid charges (quick check)
    python scripts/run_ternium_loads.py --loads 20122503142223 20082504020140
    python scripts/run_ternium_loads.py --all --impute-T-sat   also charges without T_sat (median used)
    python scripts/run_ternium_loads.py --margin 50 --heating-ramp 6

Charges already in the output csv are skipped, so an interrupted run can be resumed (--overwrite to redo).
'''

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bell_furnace.cases import build_load_case  # noqa: E402
from bell_furnace.furnace_parameters import COLD_SPOT_MARGIN, HOUR, LOAD_SCHEDULE  # noqa: E402
from bell_furnace.metrics import run_case, stack_metrics  # noqa: E402
from pipeline.data_processing import EXCEL_PATH, load_data  # noqa: E402

OUTPUT = ROOT / 'results' / 'ternium_loads'
DATA_COLUMNS = ['furnace_id', 'base_id', 'cooler_id', 'n_coils', 'load_mass', 'stack_width', 'width_max', 'thickness_mean', 'thickness_max', 'dominant_family', 'efficiency', 'valid']
SERIES = {4: '#2a78d6', 5: '#eb6834', 3: '#1baf7a'}  # categorical slots 1-3, keyed by coils per charge

# filled once per worker process by init_worker
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


def simulate_load(load_id: int, settings: dict) -> dict:
    '''
    Simulate one charge and return one row of the output table.

    Args:
        load_id:  value of RH- Carga
        settings: dict with n_r, n_z, dt_max, margin, T_sat_default and the load_schedule keys
                  heating_ramp and hold_horizon (s)

    Returns:
        Dict with load_id, the data columns, data times in h, model metrics in h and an error
        message (empty if the run worked).
    '''
    load = _LOADS.loc[load_id]
    row = {'load_id': load_id, **{c: load[c] for c in DATA_COLUMNS}}
    row.update({
        'T_sat_C': load['T_sat'] - 273.15,
        't_sat_h': load['t_sat'] / HOUR,
        't_net_h': load['t_net'] / HOUR,
        't_std_h': load['t_std'] / HOUR,
        't_net_minus_sat_h': (load['t_net'] - load['t_sat']) / HOUR,
    })
    try:
        case = build_load_case(load_id=load_id, coils_table=_COILS, loads_table=_LOADS, T_sat_default=settings['T_sat_default'], n_r=settings['n_r'], n_z=settings['n_z'], heating_ramp=settings['heating_ramp'], hold_horizon=settings['hold_horizon'])
        results = run_case(case=case, heating_only=True, dt_max=settings['dt_max'])
        m = stack_metrics(case=case, results=results, margin=settings['margin'])
    except Exception as error:  # one bad charge must not stop the batch
        row['error'] = f'{type(error).__name__}: {error}'
        return row

    row.update({
        'T_sat_model_C': case['T_hold'] - 273.15,
        'T_sat_imputed': case['T_sat_imputed'],
        'model_t_heat_cold_h': m['t_heat_cold'] / HOUR,
        'model_t_heat_mean_h': m['t_heat_mean'] / HOUR,
        'model_t_model_h': m['t_model'] / HOUR,
        'model_slowest_coil': m['slowest_coil'],
        'model_dT_max_K': m['dT_max'],
        'model_energy_GJ': m['energy'] / 1e9,
        'model_reached': m['reached'],
        'stack_problems': ' | '.join(case['problems']),
        'error': '',
    })
    for i, t in enumerate(m['t_heat_coils'], start=1):
        row[f'model_t_heat_coil{i}_h'] = t / HOUR
    return row


def plot_comparison(*, table: pd.DataFrame, path: Path) -> None:
    '''
    Plot the model heating time against the data time and against the charge mass.

    Two panels, one axis each: model against t_net - t_sat (with the 1:1 line) and model against
    the charge mass. Colour is the number of coils per charge.

    Args:
        table:  output table of this script
        path:   output png path
    '''
    ok = table[table['model_reached'].fillna(False).astype(bool)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for n, group in ok.groupby('n_coils'):
        style = {'s': 22, 'color': SERIES.get(int(n), '#888888'), 'alpha': 0.75, 'edgecolors': 'white', 'linewidths': 0.5, 'label': f'{int(n)} coils (n = {len(group)})'}
        ax1.scatter(group['t_net_minus_sat_h'], group['model_t_heat_cold_h'], **style)
        ax2.scatter(group['load_mass'] / 1000.0, group['model_t_heat_cold_h'], **style)

    if not ok.empty:
        low = min(ok['t_net_minus_sat_h'].min(), ok['model_t_heat_cold_h'].min())
        high = max(ok['t_net_minus_sat_h'].max(), ok['model_t_heat_cold_h'].max())
        ax1.plot([low, high], [low, high], color='#888888', linewidth=1.0, linestyle='--', label='1:1')
        r = np.corrcoef(ok['t_net_minus_sat_h'], ok['model_t_heat_cold_h'])[0, 1]
        ax1.set_title(f'Model heating vs data (heating + cooling), r = {r:.2f}', fontsize=10)
        r = np.corrcoef(ok['load_mass'], ok['model_t_heat_cold_h'])[0, 1]
        ax2.set_title(f'Model heating vs charge mass, r = {r:.2f}', fontsize=10)

    ax1.set_xlabel('Data: t_net - t_sat (h)')
    ax2.set_xlabel('Charge mass (t)')
    for ax in (ax1, ax2):
        ax.set_ylabel('Model: cold spot heating time (h)')
        ax.grid(True, color='#e5e5e5', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--excel', type=Path, default=EXCEL_PATH, help='path of Recocido -Tec.xlsx')
    parser.add_argument('--loads', type=int, nargs='+', default=None, help='RH- Carga values to run')
    parser.add_argument('--sample', type=int, default=None, help='run this many random charges')
    parser.add_argument('--all', action='store_true', help='include charges flagged as not valid')
    parser.add_argument('--impute-T-sat', action='store_true', help='use the median T_sat for charges without one')
    parser.add_argument('--margin', type=float, default=COLD_SPOT_MARGIN, help='cold spot margin below T_sat in K')
    parser.add_argument('--heating-ramp', type=float, default=LOAD_SCHEDULE['heating_ramp'] / HOUR, help='gas ramp to T_sat in h')
    parser.add_argument('--hold-horizon', type=float, default=LOAD_SCHEDULE['hold_horizon'] / HOUR, help='hold before the soak in h, must exceed the heating time')
    parser.add_argument('--n-r', type=int, default=11, help='radial nodes per coil (11 differs from 21 by < 0.1 h)')
    parser.add_argument('--n-z', type=int, default=15, help='axial nodes per coil')
    parser.add_argument('--dt-max', type=float, default=120.0, help='largest time step in s (120 differs from 30 by < 0.1 h)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true', help='rerun charges already in the csv')
    args = parser.parse_args()

    coils, loads = load_data(path=args.excel)
    if args.loads is not None:
        ids = list(args.loads)
    else:
        pool = loads if args.all else loads[loads['valid']]
        if not args.impute_T_sat:
            pool = pool[pool['T_sat_ok']]
        ids = list(pool.index)
        if args.sample is not None:
            ids = list(pd.Series(ids).sample(n=min(args.sample, len(ids)), random_state=args.seed))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / 'model_vs_data.csv'
    done = pd.read_csv(csv_path) if csv_path.exists() and not args.overwrite else pd.DataFrame()
    if not done.empty:
        done = done[done['error'].fillna('') == '']
        ids = [i for i in ids if i not in set(done['load_id'])]

    settings = {
        'n_r': args.n_r, 'n_z': args.n_z, 'dt_max': args.dt_max, 'margin': args.margin,
        'T_sat_default': float(loads.loc[loads['T_sat_ok'], 'T_sat'].median()) if args.impute_T_sat else None,
        'heating_ramp': args.heating_ramp * HOUR, 'hold_horizon': args.hold_horizon * HOUR,
    }
    print(f'{len(ids)} charges to run on {args.workers} workers ({len(done)} already in {csv_path.name})')

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(str(args.excel),)) as executor:
        futures = [executor.submit(simulate_load, i, settings) for i in ids]
        for k, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if k % 25 == 0 or k == len(futures):
                print(f'  {k}/{len(futures)}')
                pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(csv_path, index=False)

    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(csv_path, index=False)
    plot_comparison(table=table, path=OUTPUT / 'model_vs_data.png')

    failed = table[table['error'].fillna('') != '']
    ok = table[table['model_reached'].fillna(False).astype(bool)]
    print(f'{len(table)} charges in {csv_path}, {len(failed)} errors, {len(table) - len(failed) - len(ok)} did not reach T_sat - margin (raise --hold-horizon)')
    if not ok.empty:
        print(f"model heating (cold spot): median {ok['model_t_heat_cold_h'].median():.1f} h; data t_net - t_sat: median {ok['t_net_minus_sat_h'].median():.1f} h")
        print(f"correlation model heating vs data t_net: {np.corrcoef(ok['model_t_heat_cold_h'], ok['t_net_h'])[0, 1]:.2f}")

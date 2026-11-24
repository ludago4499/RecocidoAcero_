'''
Compare the heating of SAE 1006 and SAE 1010 under identical conditions (Ternium case).

Both runs use the same stack geometry, gas schedule, flow and contact parameters; only the steel grade
changes (every coil of the stack gets the same grade). The heating time to a target temperature is the
first time the cold spot reaches it (linear interpolation between saved times).

To reach every target the hold is long (default 60 h) and the schedule ends there: the cooling ramp is
not part of this study.

Outputs (results/comparison):
    R1_curvas_temperatura.png   cold and hot spot of the stack vs time, both grades, targets marked,
                                with the difference T_1006 - T_1010 underneath
    R2_tiempos_vs_objetivo.png  heating time vs target temperature and the difference between grades
    R3_mapas_rz.png             r-z field of the slowest coil, both grades, same colour scale, at fixed
                                times and at the time each grade reaches the main target
    R4_mapa_r_t.png             temperature vs radius and time at mid-width of the slowest coil
    tabla_tiempos.csv           targets, times of both grades and the difference
    tabla_referencia.csv        same for the Ternium reference point (coil 1#, r = 0.889 m)
    resumen.txt                 time constants and other numbers quoted in METODOLOGIA_Y_RESULTADOS.md

Usage, from the project root:
    python scripts/compare_grades.py
    python scripts/compare_grades.py --hold-hours 80 --targets 550 600 635 660
'''

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bell_furnace.cases import build_case, probe_temperature  # noqa: E402
from bell_furnace.solver import make_h_function, make_lambda_r_function, simulate_stack, time_to_reach, trapezoid_schedule  # noqa: E402

OUT = ROOT / 'results' / 'comparison'
HOUR = 3600.0
TO_C = 273.15
GRADES = ('SAE1006', 'SAE1010')
COLORS = {'SAE1006': 'tab:blue', 'SAE1010': 'tab:orange'}
DEFAULT_TARGETS_C = [500.0, 550.0, 600.0, 625.0, 635.0, 650.0, 665.0, 675.0]
MAIN_TARGET_C = 635.0  # Ternium contact reference, 1,175 F


def run_grade(*, grade: str, hold_hours: float, save_interval: float) -> dict:
    '''
    Simulate the Ternium stack with every coil of one grade, heating ramp plus a long hold.

    Args:
        grade:         'SAE1006' or 'SAE1010'
        hold_hours:    duration of the hold at the gas saturation temperature in h
        save_interval: time between saved results in s

    Returns:
        Dict with the case dict (case), results per coil (results), time in h (time_h), stack cold and
        hot spot in K (T_cold, T_hot), gas temperature in K (T_gas) and the reference probe in K (probe,
        mid-width of coil 1# at the Ternium radius).
    '''
    case = build_case(case='ternium', grade=grade)
    s = case['T_hold']
    # same ramp as the Ternium case, long hold, a short final ramp only because the schedule needs one
    schedule_time, schedule_temperature = trapezoid_schedule(T_start=case['T_initial'], T_hold=s, T_end=s - 1.0, heating_time=case['heating_time'], holding_time=hold_hours * HOUR, cooling_time=60.0)
    h_function = make_h_function(meshes=case['meshes'], furnace=case['furnace'])
    lambda_r_function = make_lambda_r_function(strip_thickness=case['strip_thickness'], contact=case['contact'], grades=case['grades'])
    results = simulate_stack(meshes=case['meshes'], T_initial=case['T_initial'], schedule_time=schedule_time, schedule_temperature=schedule_temperature, h_function=h_function, lambda_r_function=lambda_r_function, save_interval=save_interval, grades=case['grades'])

    ref = case['reference']
    probe_all = probe_temperature(mesh=case['meshes'][ref['coil']], result=results[ref['coil']], r=ref['r'])
    return {
        'case': case,
        'results': results,
        'time_h': results[0]['time'] / HOUR,
        'T_cold': np.min([r['T_min'] for r in results], axis=0),
        'T_hot': np.max([r['T_max'] for r in results], axis=0),
        'T_gas': results[0]['T_gas'],
        'probe': probe_all[:, probe_all.shape[1] // 2],
    }


def heating_times(*, run: dict, key: str, targets_c: list[float]) -> np.ndarray:
    '''
    Time at which a temperature series first reaches each target.

    Args:
        run:       dict from run_grade
        key:       series to use: 'T_cold', 'T_hot' or 'probe'
        targets_c: target temperatures in C

    Returns:
        Times in h, nan where the target is not reached within the run.
    '''
    t = run['time_h'] * HOUR
    return np.array([time_to_reach(time=t, values=run[key], target=T + TO_C) for T in targets_c]) / HOUR


def time_constant(*, run: dict, key: str = 'T_cold', window_c: tuple[float, float] = (40.0, 10.0)) -> float:
    '''
    Fit the late-hold time constant tau of T_gas - T = A exp(-t / tau).

    Only the part of the hold where T_gas - T is between window_c[0] and window_c[1] C is used, where the
    approach to the gas temperature is close to exponential.

    Args:
        run:      dict from run_grade
        key:      series to fit
        window_c: largest and smallest T_gas - T used in the fit in K

    Returns:
        tau in h, nan if fewer than 3 points fall in the window.
    '''
    gap = run['T_gas'] - run[key]
    holding = run['time_h'] > run['case']['heating_time'] / HOUR
    use = holding & (gap <= window_c[0]) & (gap >= window_c[1])
    if use.sum() < 3:
        return float('nan')
    slope, _ = np.polyfit(run['time_h'][use], np.log(gap[use]), 1)
    return float(-1.0 / slope)


def slowest_coil(run: dict) -> int:
    '''
    Index of the coil whose cold spot is lowest at the end of the heating ramp.

    Args:
        run: dict from run_grade

    Returns:
        Coil index, 0 is the bottom coil.
    '''
    k = int(np.searchsorted(run['time_h'], run['case']['heating_time'] / HOUR))
    return int(np.argmin([r['T_min'][k] for r in run['results']]))


def plot_curves(*, runs: dict, targets_c: list[float], times: dict, path: Path) -> None:
    '''
    Overlay cold and hot spot curves of both grades, mark the targets and the time each grade reaches them.

    Args:
        runs:      grade -> dict from run_grade
        targets_c: target temperatures in C
        times:     grade -> cold spot heating times in h (same order as targets_c)
        path:      output png path
    '''
    fig, (ax, ax_d) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
    first = runs[GRADES[0]]
    ax.plot(first['time_h'], first['T_gas'] - TO_C, '--', color='gray', label='Gas (condición de frontera)')
    for g in GRADES:
        r = runs[g]
        ax.plot(r['time_h'], r['T_cold'] - TO_C, color=COLORS[g], linewidth=2.0, label=f'{g} punto frío')
        ax.plot(r['time_h'], r['T_hot'] - TO_C, color=COLORS[g], linewidth=1.0, alpha=0.6, linestyle=':', label=f'{g} punto caliente')
    for T in targets_c:
        ax.axhline(T, color='black', linewidth=0.5, alpha=0.4)
        ax.text(first['time_h'][-1], T, f' {T:.0f} °C', va='center', fontsize=8)
    for g, marker in zip(GRADES, ('o', 's')):
        ok = ~np.isnan(times[g])
        ax.plot(times[g][ok], np.array(targets_c)[ok], marker, color=COLORS[g], markersize=6, markerfacecolor='white', markeredgewidth=1.5, label=f'{g} alcanza objetivo')
    # zoom around the main targets, where the two curves overlap at full scale
    inset = ax.inset_axes([0.42, 0.25, 0.33, 0.38])
    for g, marker in zip(GRADES, ('o', 's')):
        r = runs[g]
        inset.plot(r['time_h'], r['T_cold'] - TO_C, color=COLORS[g], linewidth=2.0)
        ok = ~np.isnan(times[g])
        inset.plot(times[g][ok], np.array(targets_c)[ok], marker, color=COLORS[g], markerfacecolor='white', markeredgewidth=1.5)
    zoom_t = times['SAE1006'][list(targets_c).index(MAIN_TARGET_C)] if MAIN_TARGET_C in targets_c else np.nanmedian(times['SAE1006'])
    inset.set_xlim(zoom_t - 0.5, zoom_t + 0.5)
    inset.set_ylim(MAIN_TARGET_C - 2.0, MAIN_TARGET_C + 2.0)
    inset.axhline(MAIN_TARGET_C, color='black', linewidth=0.6)
    inset.set_title(f'Zoom en {MAIN_TARGET_C:.0f} °C (±0.5 h)', fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(True, alpha=0.4)
    ax.indicate_inset_zoom(inset, edgecolor='gray')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_title('Punto frío y punto caliente de la pila, SAE 1006 vs SAE 1010 (mismas condiciones)')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc='lower right', ncol=2)

    diff = runs['SAE1006']['T_cold'] - runs['SAE1010']['T_cold']
    ax_d.plot(first['time_h'], diff, color='black')
    ax_d.axhline(0.0, color='gray', linewidth=0.8)
    ax_d.set_ylabel('T₁₀₀₆ − T₁₀₁₀\npunto frío (K)')
    ax_d.set_xlabel('Tiempo (h)')
    ax_d.grid(True, alpha=0.4)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_times(*, targets_c: list[float], times: dict, T_gas_c: float, path: Path) -> None:
    '''
    Heating time vs target temperature for both grades and their difference.

    Args:
        targets_c: target temperatures in C
        times:     grade -> cold spot heating times in h
        T_gas_c:   gas holding temperature in C (asymptote)
        path:      output png path
    '''
    targets = np.array(targets_c)
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for g, marker in zip(GRADES, ('o', 's')):
        ax.plot(targets, times[g], marker + '-', color=COLORS[g], label=g)
    ax.axvline(T_gas_c, color='gray', linestyle='--', label=f'Gas {T_gas_c:.0f} °C (t → ∞)')
    ax.set_xlabel('Temperatura objetivo del punto frío (°C)')
    ax.set_ylabel('Tiempo de calentamiento (h)')
    ax.grid(True, alpha=0.4)
    ax.legend()

    dt_min = (times['SAE1010'] - times['SAE1006']) * 60.0
    ax_d.plot(targets, dt_min, 'D-', color='black')
    ax_d.axvline(T_gas_c, color='gray', linestyle='--')
    ax_d.set_xlabel('Temperatura objetivo del punto frío (°C)')
    ax_d.set_ylabel('Δt = t₁₀₁₀ − t₁₀₀₆ (min)')
    ax_d.grid(True, alpha=0.4)
    fig.suptitle('Tiempo para que el punto frío de la pila alcance la temperatura objetivo')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rz_maps(*, runs: dict, coil: int, fixed_h: list[float], target_c: float, times_target: dict, path: Path) -> None:
    '''
    r-z temperature field of one coil for both grades with one shared colour scale, plus their difference.

    Rows: SAE 1006, SAE 1010 (shared scale) and T_1006 - T_1010 (diverging scale). The last column is the
    saved time closest to the instant each grade reaches target_c; the cold spot of the coil is marked
    with a cyan cross and the target isotherm is drawn in white.

    Args:
        runs:         grade -> dict from run_grade
        coil:         coil index to draw
        fixed_h:      times in h shown for both grades
        target_c:     main target temperature in C
        times_target: grade -> time in h the cold spot reaches target_c
        path:         output png path
    '''
    n_col = len(fixed_h) + 1
    fields = {}
    for g in GRADES:
        r, res = runs[g], runs[g]['results'][coil]
        for col, t_h in enumerate(fixed_h + [times_target[g]]):
            k = int(np.argmin(np.abs(r['time_h'] - t_h)))
            fields[(g, col)] = (res['T'][k] - TO_C, r['time_h'][k], k)
    T_min = min(f.min() for f, _, _ in fields.values())
    T_max = max(f.max() for f, _, _ in fields.values())
    step = max(5.0, 5.0 * np.ceil((T_max - T_min) / 24.0 / 5.0))
    levels = np.arange(np.floor(T_min / step) * step, np.ceil(T_max / step) * step + 0.5 * step, step)

    # difference at the same instants (target column: instant of SAE 1006)
    res6, res10 = runs['SAE1006']['results'][coil], runs['SAE1010']['results'][coil]
    diffs = [res6['T'][fields[('SAE1006', col)][2]] - res10['T'][fields[('SAE1006', col)][2]] for col in range(n_col)]
    d_max = max(np.abs(d).max() for d in diffs)
    d_levels = np.linspace(-d_max, d_max, 17)

    mesh = runs['SAE1006']['case']['meshes'][coil]
    fig, axes = plt.subplots(3, n_col, figsize=(2.3 * n_col + 1.8, 10.0), sharex=True, sharey=True, layout='constrained')
    for row, g in enumerate(GRADES):
        for col in range(n_col):
            ax = axes[row, col]
            T, t_shown, _ = fields[(g, col)]
            cs = ax.contourf(mesh['R'], mesh['Z'], T, levels=levels, cmap='inferno')
            if col == n_col - 1:
                if T.min() <= target_c <= T.max():
                    ax.contour(mesh['R'], mesh['Z'], T, levels=[target_c], colors='white', linewidths=1.5)
                iz, ir = np.unravel_index(np.argmin(T), T.shape)
                ax.plot(mesh['R'][iz, ir], mesh['Z'][iz, ir], 'x', color='cyan', markersize=10, markeredgewidth=2.5)
                ax.set_title(f'{g}, {t_shown:.2f} h\npunto frío = {T.min():.0f} °C (x)', fontsize=9, color=COLORS[g], fontweight='bold')
            else:
                ax.set_title(f'{g}, {t_shown:.0f} h', fontsize=9)
            ax.set_aspect('equal')
        axes[row, 0].set_ylabel('z (m)')
    for col in range(n_col):
        ax = axes[2, col]
        cd = ax.contourf(mesh['R'], mesh['Z'], diffs[col], levels=d_levels, cmap='RdBu_r')
        label = f'{fields[("SAE1006", col)][1]:.0f} h' if col < n_col - 1 else f'{fields[("SAE1006", col)][1]:.2f} h'
        ax.set_title(f'T₁₀₀₆ − T₁₀₁₀, {label}', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel('r (m)')
    axes[2, 0].set_ylabel('z (m)')
    fig.colorbar(cs, ax=axes[:2, :], label='Temperatura (°C), misma escala para ambos aceros', shrink=0.9)
    fig.colorbar(cd, ax=axes[2, :], label='Diferencia (K)', shrink=0.9)
    fig.suptitle(f'Campo de temperatura del rollo {coil + 1}# (el más lento). Última columna: instante en que el punto frío de la pila llega a {target_c:.0f} °C', fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rt_maps(*, runs: dict, coil: int, target_c: float, times_target: dict, path: Path) -> None:
    '''
    Temperature vs radius and time at mid-width of one coil for both grades, shared colour scale.

    Args:
        runs:         grade -> dict from run_grade
        coil:         coil index to draw
        target_c:     main target temperature in C
        times_target: grade -> time in h the cold spot reaches target_c
        path:         output png path
    '''
    maps = {}
    for g in GRADES:
        res = runs[g]['results'][coil]
        mid = res['T'].shape[1] // 2
        maps[g] = res['T'][:, mid, :] - TO_C  # (n_save, n_r)
    T_min = min(m.min() for m in maps.values())
    T_max = max(m.max() for m in maps.values())
    levels = np.arange(np.floor(T_min / 25.0) * 25.0, np.ceil(T_max / 25.0) * 25.0 + 1.0, 25.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout='constrained')
    for ax, g in zip(axes, GRADES):
        r = runs[g]
        mesh = r['case']['meshes'][coil]
        cs = ax.contourf(r['time_h'], mesh['r'], maps[g].T, levels=levels, cmap='inferno')
        ax.contour(r['time_h'], mesh['r'], maps[g].T, levels=[target_c], colors='white', linewidths=1.5)
        ax.axvline(times_target[g], color='cyan', linestyle='--', linewidth=1.2, label=f'punto frío de la pila a {target_c:.0f} °C: {times_target[g]:.2f} h')
        ax.set_title(g, color=COLORS[g], fontweight='bold')
        ax.set_xlabel('Tiempo (h)')
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('Posición radial r (m)')
    fig.colorbar(cs, ax=axes, label='Temperatura (°C)')
    fig.suptitle(f'Rollo {coil + 1}#, plano medio z = ancho/2. Isoterma blanca: {target_c:.0f} °C')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_table(*, path: Path, targets_c: list[float], times: dict) -> list[list]:
    '''
    Write the heating time table as csv.

    Args:
        path:      output csv path
        targets_c: target temperatures in C
        times:     grade -> heating times in h

    Returns:
        The table rows (without header).
    '''
    rows = []
    for i, T in enumerate(targets_c):
        t6, t10 = times['SAE1006'][i], times['SAE1010'][i]
        dt = (t10 - t6) * 60.0
        rows.append([T, T + TO_C, t6, t10, dt, 100.0 * (t10 - t6) / t6 if t6 > 0 else float('nan')])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['T_objetivo_C', 'T_objetivo_K', 't_SAE1006_h', 't_SAE1010_h', 'dt_min', 'dt_pct'])
        for row in rows:
            w.writerow([f'{row[0]:.0f}', f'{row[1]:.2f}', f'{row[2]:.3f}', f'{row[3]:.3f}', f'{row[4]:.1f}', f'{row[5]:.2f}'])
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hold-hours', type=float, default=60.0, help='hold at the gas saturation temperature in h')
    parser.add_argument('--targets', type=float, nargs='+', default=DEFAULT_TARGETS_C, help='target temperatures in C')
    parser.add_argument('--save-interval', type=float, default=300.0, help='time between saved results in s')
    args = parser.parse_args()
    targets_c = sorted(args.targets)
    main_target = MAIN_TARGET_C if MAIN_TARGET_C in targets_c else targets_c[len(targets_c) // 2]

    OUT.mkdir(parents=True, exist_ok=True)
    runs = {g: run_grade(grade=g, hold_hours=args.hold_hours, save_interval=args.save_interval) for g in GRADES}

    times = {g: heating_times(run=runs[g], key='T_cold', targets_c=targets_c) for g in GRADES}
    times_probe = {g: heating_times(run=runs[g], key='probe', targets_c=targets_c) for g in GRADES}
    rows = write_table(path=OUT / 'tabla_tiempos.csv', targets_c=targets_c, times=times)
    rows_probe = write_table(path=OUT / 'tabla_referencia.csv', targets_c=targets_c, times=times_probe)

    T_gas_c = runs['SAE1006']['case']['T_hold'] - TO_C
    i_main = targets_c.index(main_target)
    times_main = {g: times[g][i_main] for g in GRADES}
    coil = slowest_coil(runs['SAE1006'])

    plot_curves(runs=runs, targets_c=targets_c, times=times, path=OUT / 'R1_curvas_temperatura.png')
    plot_times(targets_c=targets_c, times=times, T_gas_c=T_gas_c, path=OUT / 'R2_tiempos_vs_objetivo.png')
    plot_rz_maps(runs=runs, coil=coil, fixed_h=[3.0, 8.0, 12.0, 20.0, 30.0], target_c=main_target, times_target=times_main, path=OUT / 'R3_mapas_rz.png')
    plot_rt_maps(runs=runs, coil=coil, target_c=main_target, times_target=times_main, path=OUT / 'R4_mapa_r_t.png')

    # numbers quoted in the report
    lines = [f'Gas hold {T_gas_c:.2f} C, heating ramp {runs["SAE1006"]["case"]["heating_time"] / HOUR:.1f} h, hold {args.hold_hours:.0f} h, save interval {args.save_interval:.0f} s']
    lines.append(f'Slowest coil (lowest cold spot at end of ramp): {coil + 1}#')
    for g in GRADES:
        r = runs[g]
        k = int(np.searchsorted(r['time_h'], r['case']['heating_time'] / HOUR))
        cold_coils = [int(np.argmin([res['T_min'][j] for res in r['results']])) + 1 for j in (k, len(r['time_h']) - 1)]
        lines.append(f'{g}: tau cold spot {time_constant(run=r):.2f} h, tau probe {time_constant(run=r, key="probe"):.2f} h, cold spot at end of ramp {r["T_cold"][k] - TO_C:.1f} C, hot spot {r["T_hot"][k] - TO_C:.1f} C, coil with the cold spot at end of ramp/end of run {cold_coils}')
    diff = runs['SAE1006']['T_cold'] - runs['SAE1010']['T_cold']
    j = int(np.argmax(np.abs(diff)))
    lines.append(f'max |T_cold 1006 - T_cold 1010| = {diff[j]:.2f} K at {runs["SAE1006"]["time_h"][j]:.1f} h')
    lines.append('Cold spot of the stack: T_objetivo C, t_1006 h, t_1010 h, dt min, dt %')
    lines += [f'  {r[0]:.0f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.1f}  {r[5]:.2f}' for r in rows]
    lines.append('Reference point coil 1#, r = 0.889 m, mid-width: T_objetivo C, t_1006 h, t_1010 h, dt min, dt %')
    lines += [f'  {r[0]:.0f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.1f}  {r[5]:.2f}' for r in rows_probe]
    summary = '\n'.join(lines)
    (OUT / 'resumen.txt').write_text(summary + '\n', encoding='utf-8')
    print(summary)
    print(f'Results saved in {OUT}')

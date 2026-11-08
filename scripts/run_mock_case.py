'''
Run a case (Ternium data or the four-coil case of Yang et al. 2025) using solver.py and draw three figures.

Figures (saved in results/figures/<case>):
    1_coil_histories.png       hot spot, cold spot and gas temperature of every coil
    2_temperature_fields.png   r-z temperature field of the bottom and top coil at several times
    3_coefficients.png         h per surface and lambda_r against temperature

Usage, from the project root:
    python scripts/run_mock_case.py                          Ternium case, grades as in TERNIUM_COILS
    python scripts/run_mock_case.py --grade SAE1010          Ternium case, every coil SAE 1010
    python scripts/run_mock_case.py --case paper             paper case, generic steel
'''

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bell_furnace.cases import CASES, build_case, probe_temperature  # noqa: E402
from bell_furnace.properties import STEEL_GRADES  # noqa: E402
from bell_furnace.solver import SURFACES, make_h_function, make_lambda_r_function, simulate_stack, time_to_reach  # noqa: E402

FIGURES = ROOT / 'results' / 'figures'
HOUR = 3600.0


def plot_histories(*, results: list[dict], path: Path, meshes: list[dict] | None = None, reference: dict | None = None) -> None:
    '''
    Plot gas, hot spot and cold spot temperature of every coil, one panel per coil.

    Args:
        results:   result dicts from simulate_stack, bottom coil first
        path:      output png path
        meshes:    mesh dicts, needed only with reference
        reference: None, or dict with coil, r in m and T in K; draws the temperature at that radius
                   (mid-width) and the reference value on that coil's panel
    '''
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, i in zip(axes.flat, [3, 2, 1, 0]):  # top coil in the upper-left panel
        r = results[i]
        t = r['time'] / HOUR # change to hour format
        ax.plot(t, r['T_gas'] - 273.15, '--', color='gray', label='Gas')
        ax.plot(t, r['T_max'] - 273.15, color='tab:red', label='Hot spot')
        ax.plot(t, r['T_min'] - 273.15, color='tab:blue', label='Cold spot')
        if reference is not None and i == reference['coil']:
            probe = probe_temperature(mesh=meshes[i], result=r, r=reference['r'])
            ax.plot(t, probe[:, probe.shape[1] // 2] - 273.15, color='tab:green', label=f"r = {reference['r']:.3f} m, mid-width")
            ax.axhline(reference['T'] - 273.15, color='tab:green', linestyle=':', label='Ternium reference')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f"Coil {i + 1}# ({r['grade']})")
        ax.grid(True)
    for ax in axes[1]:
        ax.set_xlabel('Time (h)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fields(*, meshes: list[dict], results: list[dict], times_h: list[float], path: Path) -> None:
    '''
    Plot the r-z temperature field of the top and bottom coil at several times.

    Args:
        meshes:  mesh dicts from build_stack, bottom coil first
        results: result dicts from simulate_stack, bottom coil first
        times_h: times to show in h
        path:    output png path
    '''
    rows = [len(results) - 1, 0]  # top coil, bottom coil

    # collect every field first so all panels share one temperature scale
    fields = {}
    for i in rows:
        r = results[i]
        for t_h in times_h:
            k = int(np.argmin(np.abs(r['time'] - t_h * HOUR)))
            fields[(i, t_h)] = r['T'][k] - 273.15
    T_min = min(T.min() for T in fields.values())
    T_max = max(T.max() for T in fields.values())
    step = max(5.0, 5.0 * np.ceil((T_max - T_min) / 20.0 / 5.0))  # about 20 bands, multiple of 5 C
    T_low = np.floor(T_min / step) * step
    T_high = np.ceil(T_max / step) * step
    if T_high <= T_low:
        T_high = T_low + step
    levels = np.arange(T_low, T_high + 0.5 * step, step)

    fig, axes = plt.subplots(len(rows), len(times_h), figsize=(2.4 * len(times_h) + 1.6, 7.2), sharex=True, sharey=True, layout='constrained')
    for row, i in enumerate(rows):
        mesh = meshes[i]
        for col, t_h in enumerate(times_h):
            ax = axes[row, col]
            cs = ax.contourf(mesh['R'], mesh['Z'], fields[(i, t_h)], levels=levels, cmap='inferno', vmin=T_low, vmax=T_high)
            ax.set_title(f"Coil {i + 1}# ({results[i]['grade']}), {t_h:g} h", fontsize=9)
            ax.set_aspect('equal')
        axes[row, 0].set_ylabel('z (m)')
    for ax in axes[-1]:
        ax.set_xlabel('r (m)')
    fig.colorbar(cs, ax=axes, label='Temperature (°C)', shrink=0.9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coefficients(*, h_function: callable, lambda_r_function: callable, path: Path) -> None:
    '''
    Plot h per surface of coil 2# against gas temperature, and lambda_r of coil 1# against node temperature.

    Args:
        h_function:        function from make_h_function
        lambda_r_function: function from make_lambda_r_function
        path:              output png path
    '''
    T = np.linspace(310.0, 1000.0, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    for surface in SURFACES:
        h = [h_function(coil=1, T_g=T_g, time=0.0)[surface] for T_g in T]
        ax1.plot(T - 273.15, h, label=surface)
    ax1.set_title('h, coil 2#')
    ax1.set_xlabel('Gas temperature (°C)')
    ax1.set_ylabel('h (W/(m² K))')
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(True)

    for T_g, label in [(500.0, 'T_g < 800 K (emulsion)'), (900.0, 'T_g ≥ 800 K (oxide)')]:
        lam = lambda_r_function(coil=0, T=T.reshape(1, -1), T_g=T_g)[0]
        ax2.plot(T - 273.15, lam, label=label)
    ax2.set_title('λr, coil 1#')
    ax2.set_xlabel('Node temperature (°C)')
    ax2.set_ylabel('λr (W/(m K))')
    ax2.set_ylim(bottom=0,top=25)
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=CASES, default='ternium')
    parser.add_argument('--grade', choices=sorted(STEEL_GRADES), default=None, help='force this grade on every coil')
    args = parser.parse_args()

    case = build_case(case=args.case, grade=args.grade, n_r=21, n_z=31)
    meshes, grades = case['meshes'], case['grades']
    h_function = make_h_function(meshes=meshes, furnace=case['furnace'])
    lambda_r_function = make_lambda_r_function(strip_thickness=case['strip_thickness'], contact=case['contact'], grades=grades)

    results = simulate_stack(meshes=meshes, T_initial=case['T_initial'], schedule_time=case['schedule_time'], schedule_temperature=case['schedule_temperature'], h_function=h_function, lambda_r_function=lambda_r_function, grades=grades)

    print(f"Case {case['name']}, gas hold {case['T_hold']:.2f} K ({case['T_hold'] - 273.15:.1f} C)")
    print('coil  grade    width [m]  mass [kg]  hot/cold at end of heating [K]  cold spot within 73 K of hold [h]')
    for i, (coil, result) in enumerate(zip(case['coils'], results)):
        k = int(np.searchsorted(result['time'], case['heating_time']))
        t_soak = time_to_reach(time=result['time'], values=result['T_min'], target=case['T_hold'] - 73.0)
        print(f"{i + 1}#    {coil['grade']:8s} {coil['width']:.3f}      {coil['mass']:8.0f}   {result['T_max'][k]:.1f} / {result['T_min'][k]:.1f}                   {t_soak / HOUR:.2f}")

    reference = case['reference']
    if reference is not None:
        i = reference['coil']
        probe = probe_temperature(mesh=meshes[i], result=results[i], r=reference['r'])
        print(f"Reference: coil {i + 1}#, r = {reference['r']:.3f} m, Ternium {reference['T']:.2f} K ({reference['T'] - 273.15:.1f} C)")
        print(f"  model at end of run: {probe[-1].min():.1f} to {probe[-1].max():.1f} K along the width; peak during the run {probe.max():.1f} K")

    folder = FIGURES / case['name']
    folder.mkdir(parents=True, exist_ok=True)
    plot_histories(results=results, path=folder / '1_coil_histories.png', meshes=meshes, reference=reference)
    plot_fields(meshes=meshes, results=results, times_h=[2.0, 6.0, 12.0, 22.0, 30.0], path=folder / '2_temperature_fields.png')
    plot_coefficients(h_function=h_function, lambda_r_function=lambda_r_function, path=folder / '3_coefficients.png')
    print(f'Figures saved in {folder}')

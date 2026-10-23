'''
Run the four-coil case of Yang et al. (2025) using solver.py and draw three figures.

Figures (saved in results/figures):
    1_coil_histories.png       hot spot, cold spot and gas temperature of every coil
    2_temperature_fields.png   r-z temperature field of the bottom and top coil at several times
    3_coefficients.png         h per surface and lambda_r against temperature

Usage, from the project root:
    python scripts/run_mock_case.py
'''

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bell_furnace.furnace_parameters import CONTACT, FURNACE, TABLE_3  # noqa: E402
from bell_furnace.mesh import build_stack, coil_inner_diameter  # noqa: E402
from bell_furnace.solver import SURFACES, make_h_function, make_lambda_r_function, simulate_stack, trapezoid_schedule  # noqa: E402

FIGURES = ROOT / 'results' / 'figures'
HOUR = 3600.0


def plot_histories(*, results: list[dict], path: Path) -> None:
    '''
    Plot gas, hot spot and cold spot temperature of every coil, one panel per coil.

    Args:
        results: result dicts from simulate_stack, bottom coil first
        path:    output png path
    '''
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, i in zip(axes.flat, [3, 2, 1, 0]):  # top coil in the upper-left panel
        r = results[i]
        t = r['time'] / HOUR # change to hour format
        ax.plot(t, r['T_gas'] - 273.15, '--', color='gray', label='Gas')
        ax.plot(t, r['T_max'] - 273.15, color='tab:red', label='Hot spot')
        ax.plot(t, r['T_min'] - 273.15, color='tab:blue', label='Cold spot')
        ax.set_title(f'Coil {i + 1}#')
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
    rows = [3, 0]  # top coil, bottom coil
    fig, axes = plt.subplots(len(rows), len(times_h), figsize=(2.4 * len(times_h) + 1.2, 7.2), sharex=True, sharey=True)
    for row, i in enumerate(rows):
        mesh, r = meshes[i], results[i]
        for col, t_h in enumerate(times_h):
            ax = axes[row, col]
            k = int(np.argmin(np.abs(r['time'] - t_h * HOUR)))
            T = r['T'][k] - 273.15
            cs = ax.contourf(mesh['R'], mesh['Z'], T, levels=12, cmap='inferno')
            fig.colorbar(cs, ax=ax, label='Temperature (°C)')            
            ax.set_title(f'Coil {i + 1}#, {t_h:g} h', fontsize=9)
            ax.set_aspect('equal')
        axes[row, 0].set_ylabel('z (m)')
    for ax in axes[-1]:
        ax.set_xlabel('r (m)')
    fig.tight_layout()
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
    ax1.legend()
    ax1.grid(True)

    for T_g, label in [(500.0, 'T_g < 800 K (emulsion)'), (900.0, 'T_g ≥ 800 K (oxide)')]:
        lam = lambda_r_function(coil=0, T=T.reshape(1, -1), T_g=T_g)[0]
        ax2.plot(T - 273.15, lam, label=label)
    ax2.set_title('λr, coil 1#')
    ax2.set_xlabel('Node temperature (°C)')
    ax2.set_ylabel('λr (W/(m K))')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    # Same setup as solver.py __main__
    coils = []
    strip_thickness = []
    for c in TABLE_3:
        inner_diameter = coil_inner_diameter(mass=c['mass'], outer_diameter=c['outer_diameter'], width=c['width'])
        coils.append({'inner_diameter': inner_diameter, 'outer_diameter': c['outer_diameter'], 'width': c['width']})
        strip_thickness.append(c['thickness'])
    meshes = build_stack(coils=coils, n_r=21, n_z=31)
    h_function = make_h_function(meshes=meshes, furnace=FURNACE)
    lambda_r_function = make_lambda_r_function(strip_thickness=strip_thickness, contact=CONTACT)

    # trapezoid scheduled temp heating
    schedule_time, schedule_temperature = trapezoid_schedule(T_start=303.15, T_hold=973.15, T_end=423.15, heating_time=12 * HOUR, holding_time=10 * HOUR, cooling_time=20 * HOUR)

    results = simulate_stack(meshes=meshes, T_initial=303.15, schedule_time=schedule_time, schedule_temperature=schedule_temperature, h_function=h_function, lambda_r_function=lambda_r_function)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_histories(results=results, path=FIGURES / '1_coil_histories.png')
    plot_fields(meshes=meshes, results=results, times_h=[2.0, 6.0, 12.0, 22.0, 30.0], path=FIGURES / '2_temperature_fields.png')
    plot_coefficients(h_function=h_function, lambda_r_function=lambda_r_function, path=FIGURES / '3_coefficients.png')
    print(f'Figures saved in {FIGURES}')

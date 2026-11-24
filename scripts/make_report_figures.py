'''
Generate every figure, table and number of the report (Ternium case, SAE 1006 vs SAE 1010).

All figures use the same style (serif font, IEEE column widths, Spanish labels) and are saved as PDF
(vector, for LaTeX) and PNG (300 dpi) in results/report. The simulations are the same as
compare_grades.py: ramp of 12 h to the gas hold temperature, long hold, no cooling.

Figures
    Methodology
        M1_geometria        stack inside the bell (to scale) and the mesh and boundary conditions of a coil
        M2_propiedades      k, cp and diffusivity of both grades vs temperature
        M3_coeficientes     h per coil and surface, h vs gas temperature, lambda_r vs k_s
        M4_curvatura        axisymmetric model vs planar (Yang et al.) model, same inputs
    Results
        R1_curvas_temperatura   cold spot and reference point vs time, both grades
        R2_tiempos_vs_objetivo  heating time vs target and difference between grades
        R3_mapas_rz             r-z field of the slowest coil, both grades, same colour scale
        R4_mapa_r_t             temperature vs radius and time at mid-width of the slowest coil
        R5_calor_por_superficie share of the absorbed heat per surface and coil
        R6_campo_pila           temperature field of the whole stack at two instants

Tables and numbers (results/report)
    tabla_tiempos.csv, tabla_referencia.csv, tabla_curvatura.csv, tabla_calor_superficies.csv, resumen.txt

Usage, from the project root:
    python scripts/make_report_figures.py
    python scripts/make_report_figures.py --targets 580 600 635 675 --hold-hours 80
'''

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from bell_furnace.cases import build_case, probe_temperature  # noqa: E402
from bell_furnace.mesh import build_stack  # noqa: E402
from bell_furnace.properties import hydrogen_properties, steel_properties  # noqa: E402
from bell_furnace.radial_conductivity import effective_radial_conductivity  # noqa: E402
from bell_furnace.solver import SURFACES, make_h_function, make_lambda_r_function, simulate_stack, time_to_reach, trapezoid_schedule  # noqa: E402
from bell_furnace.furnace_parameters import TERNIUM  # noqa: E402

OUT = ROOT / 'results' / 'report'
HOUR = 3600.0
TO_C = 273.15
GRADES = ('SAE1006', 'SAE1010')
LABEL = {'SAE1006': 'SAE 1006', 'SAE1010': 'SAE 1010'}
# categorical slots 1 and 2 of the validated palette; each grade also has its own line style and marker
COLOR = {'SAE1006': '#2a78d6', 'SAE1010': '#eb6834'}
STYLE = {'SAE1006': '-', 'SAE1010': '--'}
MARKER = {'SAE1006': 'o', 'SAE1010': 's'}
SURF_COLOR = {'outer': '#2a78d6', 'inner': '#eb6834', 'bottom': '#1baf7a', 'top': '#eda100'}
SURF_LABEL = {'outer': 'Externa', 'inner': 'Interna', 'bottom': 'Base inferior', 'top': 'Base superior'}
GAS_COLOR = '#52514e'
DEFAULT_TARGETS_C = [580.0, 600.0, 625.0, 635.0, 650.0, 665.0, 675.0]
MAIN_TARGET_C = 635.0  # Ternium contact temperature, 1,175 F
ANNEAL_RANGE_C = (580.0, 780.0)  # range given in the challenge brief
ONE_COL = 3.5  # in, IEEE column width
TWO_COL = 7.16  # in, IEEE text width

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'grid.color': '#d9d9d6',
    'lines.linewidth': 1.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


# ------------------------------------------------------------------------------------------------
# Simulations
# ------------------------------------------------------------------------------------------------

def run_case(*, grade: str, hold_hours: float, save_interval: float, geometry: str = 'axisymmetric') -> dict:
    '''
    Simulate the Ternium stack with every coil of one grade: heating ramp plus a long hold.

    Args:
        grade:         steel grade of every coil
        hold_hours:    length of the hold in h
        save_interval: time between saved results in s
        geometry:      'axisymmetric' (this model) or 'planar' (section treated as a plane, as in Yang et al.)

    Returns:
        Dict with case, meshes, results, h_function, time_h, T_cold and T_hot of the stack in K, T_gas in K,
        probe in K (Ternium reference point, mid-width) and cold_coil (coil holding the cold spot at each
        saved time).
    '''
    case = build_case(case='ternium', grade=grade)
    meshes = case['meshes'] if geometry == 'axisymmetric' else build_stack(coils=case['coils'], n_r=21, n_z=31, geometry='planar')
    s = case['T_hold']
    schedule_time, schedule_temperature = trapezoid_schedule(T_start=case['T_initial'], T_hold=s, T_end=s - 1.0, heating_time=case['heating_time'], holding_time=hold_hours * HOUR, cooling_time=60.0)
    h_function = make_h_function(meshes=meshes, furnace=case['furnace'])
    lambda_r_function = make_lambda_r_function(strip_thickness=case['strip_thickness'], contact=case['contact'], grades=case['grades'])
    results = simulate_stack(meshes=meshes, T_initial=case['T_initial'], schedule_time=schedule_time, schedule_temperature=schedule_temperature, h_function=h_function, lambda_r_function=lambda_r_function, save_interval=save_interval, grades=case['grades'])

    ref = case['reference']
    probe = probe_temperature(mesh=meshes[ref['coil']], result=results[ref['coil']], r=ref['r'])
    T_min = np.array([r['T_min'] for r in results])
    return {
        'grade': grade,
        'geometry': geometry,
        'case': case,
        'meshes': meshes,
        'results': results,
        'h_function': h_function,
        'time_h': results[0]['time'] / HOUR,
        'T_cold': T_min.min(axis=0),
        'T_hot': np.max([r['T_max'] for r in results], axis=0),
        'T_gas': results[0]['T_gas'],
        'probe': probe[:, probe.shape[1] // 2],
        'cold_coil': T_min.argmin(axis=0),
    }


def reach_times(*, run: dict, key: str, targets_c: list[float]) -> np.ndarray:
    '''
    Time at which a temperature series first reaches each target.

    Args:
        run:       dict from run_case
        key:       'T_cold', 'T_hot' or 'probe'
        targets_c: target temperatures in C

    Returns:
        Times in h, nan where the target is not reached.
    '''
    t = run['time_h'] * HOUR
    return np.array([time_to_reach(time=t, values=run[key], target=T + TO_C) for T in targets_c]) / HOUR


def nearest_index(*, run: dict, t_h: float) -> int:
    '''
    Index of the saved time closest to t_h.

    Args:
        run: dict from run_case
        t_h: time in h

    Returns:
        Index into the saved times.
    '''
    return int(np.argmin(np.abs(run['time_h'] - t_h)))


def time_constant(*, run: dict, key: str = 'T_cold', window: tuple[float, float] = (40.0, 10.0)) -> float:
    '''
    Late-hold time constant tau of T_gas - T = A exp(-t / tau).

    Args:
        run:    dict from run_case
        key:    series to fit
        window: largest and smallest T_gas - T in K used in the fit

    Returns:
        tau in h, nan if fewer than 3 points fall in the window.
    '''
    gap = run['T_gas'] - run[key]
    use = (run['time_h'] > run['case']['heating_time'] / HOUR) & (gap <= window[0]) & (gap >= window[1])
    if use.sum() < 3:
        return float('nan')
    slope, _ = np.polyfit(run['time_h'][use], np.log(gap[use]), 1)
    return float(-1.0 / slope)


def enthalpy_function(grade: str):
    '''
    Build the specific enthalpy rho * integral of cp dT of a grade, per unit volume.

    Args:
        grade: steel grade

    Returns:
        Function of T in K returning the volumetric enthalpy in J/m3 relative to 293.15 K.
    '''
    T_grid = np.linspace(293.15, 1300.0, 2000)
    rho, cp, _ = steel_properties(T_grid, grade=grade)
    H = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] * cp[1:] + rho[:-1] * cp[:-1]) * np.diff(T_grid))])
    return lambda T: np.interp(T, T_grid, H)


def heat_by_surface(*, run: dict, t_end_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Integrate the heat that enters each surface of each coil from t = 0 to t_end_h.

    Q_s = 2 pi * integral of h_s(T_g) * sum(A_s * (T_g - T)) dt, with the per-radian areas of the mesh.
    It also returns the change of stored energy of each coil as an energy-conservation check.

    Args:
        run:     dict from run_case (axisymmetric)
        t_end_h: end of the integration in h

    Returns:
        Tuple (Q, stored, error): Q in J with shape (n_coils, 4) in the order of SURFACES, stored energy
        change in J per coil, and the relative difference sum(Q) / stored - 1 per coil.
    '''
    t = run['time_h'] * HOUR
    k_end = int(np.searchsorted(run['time_h'], t_end_h))
    n = len(run['results'])
    Q = np.zeros((n, len(SURFACES)))
    stored = np.zeros(n)
    H = enthalpy_function(run['grade'])
    for i, (mesh, res) in enumerate(zip(run['meshes'], run['results'])):
        rates = np.zeros((k_end + 1, len(SURFACES)))
        for k in range(k_end + 1):
            T_g = float(res['T_gas'][k])
            h = run['h_function'](coil=i, T_g=T_g, time=t[k])
            for j, s in enumerate(SURFACES):
                rates[k, j] = h[s] * np.sum(mesh[f'area_{s}'] * (T_g - res['T'][k]))
        Q[i] = 2.0 * np.pi * np.trapezoid(rates, t[:k_end + 1], axis=0)
        stored[i] = 2.0 * np.pi * np.sum(mesh['volume'] * (H(res['T'][k_end]) - H(res['T'][0])))
    return Q, stored, Q.sum(axis=1) / stored - 1.0


def save(fig: plt.Figure, name: str) -> None:
    '''
    Save a figure as PDF and PNG in OUT and close it.

    Args:
        fig:  figure
        name: file name without extension
    '''
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png')
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# Methodology figures
# ------------------------------------------------------------------------------------------------

def plot_geometry(*, run: dict, t_main_h: float) -> None:
    '''
    M1: (a) stack inside the bell to scale, (b) mesh and boundary conditions of one coil.

    Args:
        run:      SAE 1006 run, used for the coil sizes and the cold spot at t_main_h
        t_main_h: time in h at which the stack cold spot reaches the main target
    '''
    coils = run['case']['coils']
    furnace = run['case']['furnace']
    gap, top_gap = furnace['plate_gap'], furnace['top_gap']
    R_bell = 0.5 * TERNIUM['bell_diameter']
    H_op = TERNIUM['operating_height']

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO_COL, 4.3), gridspec_kw={'width_ratios': [1.25, 1.0]})

    # (a) full section, both halves
    ax.add_patch(Rectangle((-R_bell, 0.0), 2 * R_bell, H_op, fill=False, edgecolor=GAS_COLOR, linewidth=1.0, linestyle=(0, (4, 2))))
    z0 = 0.0
    k = nearest_index(run=run, t_h=t_main_h)
    cold_i = int(run['cold_coil'][k])
    for i, c in enumerate(coils):
        r_in, r_out, w = 0.5 * c['inner_diameter'], 0.5 * c['outer_diameter'], c['width']
        for sign in (-1, 1):
            x = r_in if sign > 0 else -r_out
            ax.add_patch(Rectangle((x, z0), r_out - r_in, w, facecolor='#c9d6e8', edgecolor='#33475b', linewidth=0.7))
        ax.text(-0.5 * (r_in + r_out), z0 + 0.5 * w, f'{i + 1}#', ha='center', va='center', fontsize=8, fontweight='bold', color='#1b2a3a')
        if i < len(coils) - 1:
            ax.add_patch(Rectangle((-r_out, z0 + w), 2 * r_out, gap, facecolor='#e6e4de', edgecolor='#8c8a84', linewidth=0.5, hatch='////'))
        if i == cold_i:
            res, mesh = run['results'][i], run['meshes'][i]
            jz, jr = np.unravel_index(np.argmin(res['T'][k]), res['T'][k].shape)
            ax.plot(mesh['r'][jr], z0 + mesh['z'][jz], marker='x', color='#b3261e', markersize=7, markeredgewidth=1.6, linestyle='none', label=f'Punto frío de la pila ({i + 1}#)')
        z0 += w + gap
    ref = run['case']['reference']
    ax.plot(ref['r'], 0.5 * coils[ref['coil']]['width'], marker='o', color='#008300', markersize=5, linestyle='none', label=r'Referencia Ternium ($r$ = 0.889 m)')
    ax.axvline(0.0, color='black', linewidth=0.6, linestyle=(0, (8, 3, 1, 3)))
    ax.text(0.0, H_op + 0.08, f'Campana, $D$ = {TERNIUM["bell_diameter"]:.3f} m', fontsize=7, ha='center', va='bottom', color=GAS_COLOR, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    extra = [Line2D([], [], color='black', linewidth=0.6, linestyle=(0, (8, 3, 1, 3)), label='Eje de simetría'), Patch(facecolor='#e6e4de', edgecolor='#8c8a84', hatch='////', label='Placa convectora (espesor estimado)')]
    ax.text(0.0, z0 - gap + 0.5 * top_gap, r'Gas H$_2$ a $T_g(t)$', fontsize=7, ha='center', va='center', color=GAS_COLOR, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
    ax.annotate('', xy=(R_bell + 0.12, 0.0), xytext=(R_bell + 0.12, H_op), arrowprops={'arrowstyle': '<->', 'linewidth': 0.6})
    ax.text(R_bell + 0.17, 0.5 * H_op, f'{H_op:.2f} m', rotation=90, va='center', fontsize=7)
    c0 = coils[0]
    ax.annotate('', xy=(-0.5 * c0['inner_diameter'], -0.18), xytext=(0.5 * c0['inner_diameter'], -0.18), arrowprops={'arrowstyle': '<->', 'linewidth': 0.6})
    ax.text(0.0, -0.32, f'$D_i$ = {c0["inner_diameter"]:.3f} m', ha='center', va='top', fontsize=7)
    ax.annotate('', xy=(-0.5 * c0['outer_diameter'], -0.55), xytext=(0.5 * c0['outer_diameter'], -0.55), arrowprops={'arrowstyle': '<->', 'linewidth': 0.6})
    ax.text(0.0, -0.69, f'$D_e$ = {c0["outer_diameter"]:.3f} m', ha='center', va='top', fontsize=7)
    ax.set_xlim(-R_bell - 0.1, R_bell + 0.4)
    ax.set_ylim(-0.95, H_op + 0.4)
    ax.set_aspect('equal')
    ax.set_xlabel('$r$ (m)')
    ax.set_ylabel('$z$ (m)')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, loc='upper left', bbox_to_anchor=(-0.05, -0.12), fontsize=7, ncol=1, handletextpad=0.4)
    ax.set_title('(a) Carga de cuatro rollos dentro de la campana (a escala)', loc='left')

    # (b) one coil, half-section, with nodes and boundary conditions
    mesh = run['meshes'][cold_i]
    r_in, r_out, w = mesh['r_in'], mesh['r_out'], mesh['width']
    bx.add_patch(Rectangle((r_in, 0.0), r_out - r_in, w, facecolor='#eef2f7', edgecolor='#33475b', linewidth=0.8))
    bx.plot(mesh['R'], mesh['Z'], '.', color='#33475b', markersize=1.6)
    pad = 0.06
    bx.annotate(r'$h_{\mathrm{int}}(T_g-T)$', xy=(r_in, 0.5 * w), xytext=(r_in - pad, 0.5 * w), ha='right', va='center', fontsize=7, arrowprops={'arrowstyle': '->', 'linewidth': 0.6})
    bx.annotate(r'$h_{\mathrm{ext}}(T_g-T)$', xy=(r_out, 0.5 * w), xytext=(r_out + pad, 0.5 * w), ha='left', va='center', fontsize=7, arrowprops={'arrowstyle': '->', 'linewidth': 0.6})
    bx.annotate(r'$h_{\mathrm{sup}}(T_g-T)$', xy=(0.5 * (r_in + r_out), w), xytext=(0.5 * (r_in + r_out), w + pad), ha='center', va='bottom', fontsize=7, arrowprops={'arrowstyle': '->', 'linewidth': 0.6})
    bx.annotate(r'$h_{\mathrm{inf}}(T_g-T)$' + '\n(adiabática en el rollo 1#)', xy=(0.5 * (r_in + r_out), 0.0), xytext=(0.5 * (r_in + r_out), -pad), ha='center', va='top', fontsize=7, arrowprops={'arrowstyle': '->', 'linewidth': 0.6})
    bx.text(0.5 * (r_in + r_out), 0.72 * w, r'$k_r$ (radial efectiva)' + '\n' + r'$k_z = k_s$ (axial)', ha='center', va='center', fontsize=7, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
    bx.text(0.5 * (r_in + r_out), 0.25 * w, f'Malla: {mesh["n_r"]} × {mesh["n_z"]} nodos', ha='center', va='center', fontsize=7, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
    bx.set_xlim(r_in - 0.42, r_out + 0.32)
    bx.set_ylim(-0.3, w + 0.2)
    bx.set_aspect('equal')
    bx.set_xlabel('$r$ (m)')
    bx.set_ylabel('$z$ (m)')
    bx.set_title('(b) Dominio $(r,z)$ de un rollo y condiciones de frontera', loc='left')
    fig.tight_layout()
    save(fig, 'M1_geometria')


def plot_properties() -> None:
    '''
    M2: thermal conductivity, specific heat and diffusivity of both grades vs temperature.
    '''
    T = np.linspace(303.15, 1173.15, 400)
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.3))
    for g in GRADES:
        rho, cp, k = steel_properties(T, grade=g)
        alpha = k / (rho * cp) * 1e6
        kw = {'color': COLOR[g], 'linestyle': STYLE[g], 'label': LABEL[g]}
        axes[0].plot(T - TO_C, k, **kw)
        axes[1].plot(T - TO_C, cp, **kw)
        axes[2].plot(T - TO_C, alpha, **kw)
    labels = [r'$k_s$ (W m$^{-1}$ K$^{-1}$)', r'$c_p$ (J kg$^{-1}$ K$^{-1}$)', r'$\alpha$ (mm$^2$ s$^{-1}$)']
    titles = ['(a) Conductividad térmica', '(b) Calor específico (EN 1993-1-2)', '(c) Difusividad térmica']
    for ax, lab, tit in zip(axes, labels, titles):
        ax.axvspan(*ANNEAL_RANGE_C, color='#ebe9e3', zorder=0, linewidth=0)
        ax.set_xlabel('Temperatura (°C)')
        ax.set_ylabel(lab)
        ax.set_title(tit, loc='left')
        ax.grid(True)
    axes[1].text(0.5 * sum(ANNEAL_RANGE_C), axes[1].get_ylim()[1] * 0.97, 'Intervalo de\nrecocido', ha='center', va='top', fontsize=6.5, color=GAS_COLOR)
    axes[1].text(0.5, 0.35, 'Curvas idénticas\npara ambos grados', transform=axes[1].transAxes, ha='center', fontsize=6.5, color=GAS_COLOR)
    axes[0].legend(loc='lower left')
    fig.tight_layout()
    save(fig, 'M2_propiedades')


def plot_coefficients(*, run: dict) -> None:
    '''
    M3: (a) h of every surface and coil at the hold temperature, (b) h of the slowest coil vs gas
    temperature, (c) effective radial conductivity vs steel conductivity for both grades.

    Args:
        run: SAE 1006 run (h does not depend on the grade)
    '''
    hf = run['h_function']
    T_hold = run['case']['T_hold']
    n = len(run['meshes'])
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.4))

    ax = axes[0]
    width = 0.2
    x = np.arange(n)
    order = ['outer', 'inner', 'bottom', 'top']
    for j, s in enumerate(order):
        vals = [hf(coil=i, T_g=T_hold, time=0.0)[s] for i in range(n)]
        ax.bar(x + (j - 1.5) * width, vals, width=width * 0.9, color=SURF_COLOR[s], label=SURF_LABEL[s])
    ax.set_xticks(x, [f'{i + 1}#' for i in range(n)])
    ax.set_xlabel('Rollo')
    ax.set_ylabel(r'$h_{\mathrm{eff}}$ (W m$^{-2}$ K$^{-1}$)')
    ax.set_title(f'(a) $h_{{\\mathrm{{eff}}}}$ a $T_g$ = {T_hold - TO_C:.0f} °C', loc='left')
    ax.grid(True, axis='y')
    ax.legend(loc='upper right', fontsize=6.5)

    ax = axes[1]
    T_g = np.linspace(303.15, T_hold, 300)
    i = 2  # coil 3#, the slowest
    for s in order:
        vals = [hf(coil=i, T_g=T, time=0.0)[s] for T in T_g]
        ls = (0, (3, 1.5)) if s == 'inner' else '-'
        ax.plot(T_g - TO_C, vals, color=SURF_COLOR[s], linestyle=ls, label=SURF_LABEL[s])
    ax.axvline(800.0 - TO_C, color=GAS_COLOR, linewidth=0.6, linestyle=':')
    ax.text(800.0 - TO_C + 8, ax.get_ylim()[1] * 0.95, r'800 K: $h_{\mathrm{eff}}=\xi h$', fontsize=6.5, va='top', color=GAS_COLOR)
    ax.set_xlabel('$T_g$ (°C)')
    ax.set_ylabel(r'$h_{\mathrm{eff}}$ (W m$^{-2}$ K$^{-1}$)')
    ax.set_title('(b) Rollo 3#, colores como en (a)', loc='left')
    ax.grid(True)
    ax.text(300, 51, 'Externa = interna', fontsize=6.5, color=GAS_COLOR)

    ax = axes[2]
    T = np.linspace(303.15, T_hold, 300)
    case = run['case']
    for g in GRADES:
        _, _, k_s = steel_properties(T, grade=g)
        _, _, k_g, _ = hydrogen_properties(T)
        lam = effective_radial_conductivity(T_m=T, T_g=T_hold, k_s=k_s, k_g=k_g, s=case['strip_thickness'][0], **case['contact'])
        ax.plot(T - TO_C, k_s, color=COLOR[g], linestyle=STYLE[g], linewidth=0.9, alpha=0.6)
        ax.plot(T - TO_C, lam, color=COLOR[g], linestyle=STYLE[g], label=LABEL[g])
    ax.text(0.97, 0.93, r'$k_z = k_s$ (acero)', transform=ax.transAxes, ha='right', va='top', fontsize=6.5, color=GAS_COLOR)
    ax.text(0.97, 0.18, r'$k_r$ (Ecs. 9–17)', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5, color=GAS_COLOR)
    ax.set_xlabel('Temperatura del acero (°C)')
    ax.set_ylabel(r'$k$ (W m$^{-1}$ K$^{-1}$)')
    ax.set_title('(c) Conductividad axial y radial', loc='left')
    ax.set_ylim(0, None)
    ax.grid(True)
    ax.legend(loc='center left', fontsize=6.5, bbox_to_anchor=(0.0, 0.42))
    fig.tight_layout()
    save(fig, 'M3_coeficientes')


def plot_curvature(*, axi: dict, planar: dict, targets_c: list[float]) -> list[list]:
    '''
    M4: cold spot of the stack with the axisymmetric and the planar formulation, same inputs.

    Args:
        axi:       SAE 1006 axisymmetric run
        planar:    SAE 1006 planar run
        targets_c: target temperatures in C

    Returns:
        Table rows: target in C, time axisymmetric in h, time planar in h, difference in h and %.
    '''
    t_a = reach_times(run=axi, key='T_cold', targets_c=targets_c)
    t_p = reach_times(run=planar, key='T_cold', targets_c=targets_c)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO_COL, 2.4))
    ax.plot(axi['time_h'], axi['T_gas'] - TO_C, color=GAS_COLOR, linestyle=(0, (4, 2)), linewidth=1.0, label='Gas')
    ax.plot(axi['time_h'], axi['T_cold'] - TO_C, color=COLOR['SAE1006'], label=r'Axisimétrico $(r,z)$, este modelo')
    ax.plot(planar['time_h'], planar['T_cold'] - TO_C, color='#6250d6', linestyle=(0, (3, 1.5)), label=r'Plano $(r,z)$, sin ponderar por $r$')
    ax.set_xlabel('Tiempo (h)')
    ax.set_ylabel('Temperatura del punto frío (°C)')
    ax.set_title('(a) Punto frío de la pila, SAE 1006', loc='left')
    ax.grid(True)
    ax.legend(loc='lower right')
    bx.plot(targets_c, t_a, marker='o', markersize=4, color=COLOR['SAE1006'], label='Axisimétrico')
    bx.plot(targets_c, t_p, marker='D', markersize=4, color='#6250d6', linestyle=(0, (3, 1.5)), label='Plano')
    for T, a, p in zip(targets_c, t_a, t_p):
        if T in (targets_c[0], MAIN_TARGET_C, targets_c[-1]):
            bx.annotate(f'{p - a:+.1f} h'.replace('-', '−'), xy=(T, max(a, p)), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=6.5)
    bx.set_xlabel('Temperatura objetivo (°C)')
    bx.set_ylabel('Tiempo de calentamiento (h)')
    bx.set_title('(b) Tiempo al objetivo; etiqueta: plano − axisimétrico', loc='left')
    bx.grid(True)
    bx.legend(loc='upper left')
    fig.tight_layout()
    save(fig, 'M4_curvatura')
    return [[T, a, p, p - a, 100.0 * (p - a) / a] for T, a, p in zip(targets_c, t_a, t_p)]


# ------------------------------------------------------------------------------------------------
# Result figures
# ------------------------------------------------------------------------------------------------

def plot_curves(*, runs: dict, targets_c: list[float], times: dict, probe_times: dict) -> None:
    '''
    R1: (a) gas, cold spot of the stack and reference point vs time for both grades, (b) difference.

    Args:
        runs:        grade -> run
        targets_c:   target temperatures in C
        times:       grade -> cold spot times in h
        probe_times: grade -> reference point times in h
    '''
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(TWO_COL, 4.4), sharex=True, gridspec_kw={'height_ratios': [3, 1.1]})
    r0 = runs['SAE1006']
    ax.plot(r0['time_h'], r0['T_gas'] - TO_C, color=GAS_COLOR, linestyle=(0, (4, 2)), linewidth=1.0, label='Gas $T_g(t)$')
    for g in GRADES:
        r = runs[g]
        ax.plot(r['time_h'], r['T_cold'] - TO_C, color=COLOR[g], linestyle=STYLE[g], label=f'{LABEL[g]}, punto frío de la pila')
        ax.plot(r['time_h'], r['probe'] - TO_C, color=COLOR[g], linestyle=STYLE[g], linewidth=0.8, alpha=0.55, label=f'{LABEL[g]}, referencia Ternium')
        ax.plot(times[g], targets_c, linestyle='none', marker=MARKER[g], markersize=4, markerfacecolor='white', markeredgecolor=COLOR[g], markeredgewidth=1.0)
    for T in targets_c:
        ax.axhline(T, color='#d9d7d1', linewidth=0.4, zorder=0)
    for T in (targets_c[0], MAIN_TARGET_C, targets_c[-1]):
        ax.text(r0['time_h'][-1] + 0.8, T, f'{T:.0f} °C', va='center', fontsize=6.5, color=GAS_COLOR)
    t_main = times['SAE1006'][targets_c.index(MAIN_TARGET_C)]
    ax.annotate(f'{MAIN_TARGET_C:.0f} °C: {times["SAE1006"][targets_c.index(MAIN_TARGET_C)]:.2f} h (SAE 1006)\n{" " * 12}{times["SAE1010"][targets_c.index(MAIN_TARGET_C)]:.2f} h (SAE 1010)', xy=(t_main, MAIN_TARGET_C), xytext=(t_main + 4, 470), fontsize=7, arrowprops={'arrowstyle': '->', 'linewidth': 0.6})
    ax.set_ylabel('Temperatura (°C)')
    ax.set_ylim(0, 710)
    ax.set_title('(a) Evolución de la temperatura, mismas condiciones para ambos aceros', loc='left')
    ax.grid(True, axis='x')
    ax.legend(loc='lower right', ncol=1, fontsize=6.5)
    for key, lw, lab in (('T_cold', 1.2, 'Punto frío de la pila'), ('probe', 0.9, 'Referencia Ternium')):
        d = runs['SAE1006'][key] - runs['SAE1010'][key]
        bx.plot(r0['time_h'], d, color='black', linewidth=lw, linestyle='-' if key == 'T_cold' else (0, (3, 1.5)), label=lab)
    bx.axhline(0.0, color=GAS_COLOR, linewidth=0.5)
    bx.set_xlabel('Tiempo (h)')
    bx.set_ylabel(r'$T_{1006}-T_{1010}$ (K)')
    bx.set_title('(b) Diferencia de temperatura entre aceros', loc='left')
    bx.grid(True)
    bx.legend(loc='upper right', fontsize=6.5)
    fig.tight_layout()
    save(fig, 'R1_curvas_temperatura')


def plot_times(*, targets_c: list[float], times: dict, probe_times: dict, T_gas_c: float) -> None:
    '''
    R2: heating time vs target and the difference between grades.

    Args:
        targets_c:   target temperatures in C
        times:       grade -> cold spot times in h
        probe_times: grade -> reference point times in h
        T_gas_c:     gas hold temperature in C
    '''
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO_COL, 2.5))
    for g in GRADES:
        ax.plot(targets_c, times[g], marker=MARKER[g], color=COLOR[g], linestyle=STYLE[g], markersize=4, label=f'{LABEL[g]}, punto frío')
        ax.plot(targets_c, probe_times[g], marker=MARKER[g], color=COLOR[g], linestyle=STYLE[g], markersize=4, markerfacecolor='white', linewidth=0.8, label=f'{LABEL[g]}, referencia')
    ax.axvline(T_gas_c, color=GAS_COLOR, linestyle=(0, (4, 2)), linewidth=0.8)
    ax.text(T_gas_c - 1.5, ax.get_ylim()[1] * 0.97, f'$T_g$ = {T_gas_c:.0f} °C', rotation=90, ha='right', va='top', fontsize=6.5, color=GAS_COLOR)
    ax.set_xlabel('Temperatura objetivo (°C)')
    ax.set_ylabel('Tiempo de calentamiento (h)')
    ax.set_title('(a) Tiempo para alcanzar el objetivo', loc='left')
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=6.5)
    dt = (np.array(times['SAE1010']) - np.array(times['SAE1006'])) * 60.0
    dp = (np.array(probe_times['SAE1010']) - np.array(probe_times['SAE1006'])) * 60.0
    bx.plot(targets_c, dt, marker='o', color='black', markersize=4, label='Punto frío de la pila')
    bx.plot(targets_c, dp, marker='o', color='black', markersize=4, markerfacecolor='white', linestyle=(0, (3, 1.5)), linewidth=0.9, label='Referencia Ternium')
    for T, v in zip(targets_c, dt):
        bx.annotate(f'{v:.1f}', xy=(T, v), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=6)
    bx.axhline(0.0, color=GAS_COLOR, linewidth=0.5)
    bx.set_xlabel('Temperatura objetivo (°C)')
    bx.set_ylabel(r'$\Delta t = t_{1010}-t_{1006}$ (min)')
    bx.set_title('(b) Diferencia entre aceros', loc='left')
    bx.grid(True)
    bx.legend(loc='lower right', fontsize=6.5)
    fig.tight_layout()
    save(fig, 'R2_tiempos_vs_objetivo')


def plot_rz_maps(*, runs: dict, coil: int, fixed_h: list[float], times_main: dict) -> None:
    '''
    R3: r-z temperature field of one coil for both grades on one colour scale, plus the difference.

    Args:
        runs:       grade -> run
        coil:       coil index
        fixed_h:    fixed times in h; a last column is added at the time each grade reaches the main target
        times_main: grade -> time in h the stack cold spot reaches the main target
    '''
    n_col = len(fixed_h) + 1
    fields, diff = {}, []
    for g in GRADES:
        r = runs[g]
        ks = [nearest_index(run=r, t_h=t) for t in fixed_h] + [nearest_index(run=r, t_h=times_main[g])]
        fields[g] = [(r['time_h'][k], r['results'][coil]['T'][k] - TO_C) for k in ks]
    for c in range(n_col):
        diff.append(fields['SAE1006'][c][1] - fields['SAE1010'][c][1])
    levels = np.arange(30.0, 720.0, 30.0)
    dmax = max(np.abs(d).max() for d in diff)
    dlev = np.linspace(-dmax, dmax, 17)
    mesh = runs['SAE1006']['meshes'][coil]

    fig, axes = plt.subplots(3, n_col, figsize=(TWO_COL, 5.6), sharex=True, sharey=True)
    for row, g in enumerate(GRADES):
        for c in range(n_col):
            t, T = fields[g][c]
            ax = axes[row, c]
            cs = ax.contourf(mesh['R'], mesh['Z'], T, levels=levels, cmap='inferno', extend='neither')
            if c == n_col - 1:
                cl = ax.contour(mesh['R'], mesh['Z'], T, levels=[640.0, 645.0, 650.0], colors='#3a3a3a', linewidths=0.5)
                ax.clabel(cl, fmt='%.0f', fontsize=5.5, inline=True)
                ax.set_title(f'{LABEL[g]}, {t:.2f} h\npunto frío = {MAIN_TARGET_C:.0f} °C', fontsize=7, color=COLOR[g], fontweight='bold')
            else:
                ax.set_title(f'{LABEL[g]}, {t:.0f} h', fontsize=7)
            jz, jr = np.unravel_index(np.argmin(T), T.shape)
            ax.plot(mesh['r'][jr], mesh['z'][jz], marker='x', color='#00c8ff', markersize=5, markeredgewidth=1.2)
            ax.set_aspect('equal')
    for c in range(n_col):
        ax = axes[2, c]
        dc = ax.contourf(mesh['R'], mesh['Z'], diff[c], levels=dlev, cmap='RdBu_r')
        ax.set_title(r'$T_{1006}-T_{1010}$, ' + f'{fields["SAE1006"][c][0]:.{2 if c == n_col - 1 else 0}f} h', fontsize=7)
        ax.set_aspect('equal')
    for ax in axes[:, 0]:
        ax.set_ylabel('$z$ (m)')
    for ax in axes[-1]:
        ax.set_xlabel('$r$ (m)')
        ax.set_xticks([0.3, 0.6, 0.9])
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.915, 0.39, 0.015, 0.57))
    fig.colorbar(cs, cax=cax, label='Temperatura (°C), misma escala', ticks=np.arange(30, 720, 90))
    dax = fig.add_axes((0.915, 0.05, 0.015, 0.27))
    fig.colorbar(dc, cax=dax, label='Diferencia (K)', format='%.1f', ticks=np.linspace(-dmax, dmax, 5))
    save(fig, 'R3_mapas_rz')


def plot_rt_maps(*, runs: dict, coil: int, targets_c: list[float], times_main: dict) -> None:
    '''
    R4: temperature vs radius and time at mid-width of one coil, both grades, one colour scale.

    Args:
        runs:       grade -> run
        coil:       coil index
        targets_c:  target temperatures in C (first, main and last are drawn as isotherms)
        times_main: grade -> time in h the stack cold spot reaches the main target
    '''
    levels = np.arange(30.0, 720.0, 30.0)
    iso = sorted({targets_c[0], MAIN_TARGET_C, targets_c[-1]})
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.6), sharey=True)
    for ax, g in zip(axes, GRADES):
        r = runs[g]
        mesh = r['meshes'][coil]
        T = r['results'][coil]['T']
        M = T[:, T.shape[1] // 2, :] - TO_C
        cs = ax.contourf(r['time_h'], mesh['r'], M.T, levels=levels, cmap='inferno')
        cl = ax.contour(r['time_h'], mesh['r'], M.T, levels=iso, colors='white', linewidths=0.7)
        ax.clabel(cl, fmt='%.0f °C', fontsize=6, inline=True)
        ax.axvline(times_main[g], color='#00c8ff', linestyle=(0, (4, 2)), linewidth=0.9)
        ax.text(times_main[g] + 0.8, mesh['r'][-1] - 0.03, f'{times_main[g]:.2f} h', color='#00c8ff', fontsize=6.5, va='top')
        ax.set_title(f'{LABEL[g]}', loc='left', color=COLOR[g], fontweight='bold')
        ax.set_xlabel('Tiempo (h)')
    axes[0].set_ylabel('Posición radial $r$ (m)')
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.915, 0.18, 0.015, 0.72))
    fig.colorbar(cs, cax=cax, label='Temperatura (°C)', ticks=np.arange(30, 720, 90))
    save(fig, 'R4_mapa_r_t')


def plot_heat_share(*, Q: np.ndarray) -> None:
    '''
    R5: share of the absorbed heat per surface for each coil (stacked horizontal bars).

    Args:
        Q: heat in J per coil and surface, order of SURFACES
    '''
    order = ['outer', 'inner', 'bottom', 'top']
    idx = [SURFACES.index(s) for s in order]
    share = 100.0 * Q[:, idx] / Q.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(ONE_COL, 1.9))
    y = np.arange(len(Q))
    left = np.zeros(len(Q))
    for j, s in enumerate(order):
        ax.barh(y, share[:, j], left=left, height=0.62, color=SURF_COLOR[s], edgecolor='white', linewidth=1.0, label=SURF_LABEL[s])
        for i in range(len(Q)):
            if share[i, j] >= 6.0:
                ax.text(left[i] + 0.5 * share[i, j], i, f'{share[i, j]:.0f}', ha='center', va='center', fontsize=6.5, color='white' if s in ('outer', 'inner') else 'black')
        left += share[:, j]
    ax.set_yticks(y, [f'{i + 1}#' for i in y])
    ax.set_ylabel('Rollo')
    ax.set_xlabel('Fracción del calor absorbido (%)')
    ax.set_xlim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=6.5, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout()
    save(fig, 'R5_calor_por_superficie')


def plot_stack_field(*, run: dict, t_list_h: list[float], titles: list[str]) -> None:
    '''
    R6: temperature field of every coil of the stack, drawn at its height, at several instants.

    Args:
        run:      run to draw
        t_list_h: instants in h
        titles:   panel titles
    '''
    gap = run['case']['furnace']['plate_gap']
    levels = np.arange(30.0, 720.0, 30.0)
    fig, axes = plt.subplots(1, len(t_list_h), figsize=(ONE_COL, 3.4), sharey=True, layout='constrained')
    for ax, t_h, tit in zip(axes, t_list_h, titles):
        k = nearest_index(run=run, t_h=t_h)
        z0 = 0.0
        T_all = [res['T'][k] for res in run['results']]
        T_low = min(T.min() for T in T_all)
        for i, (mesh, T) in enumerate(zip(run['meshes'], T_all)):
            cs = ax.contourf(mesh['R'], mesh['Z'] + z0, T - TO_C, levels=levels, cmap='inferno')
            ax.text(mesh['r_out'] + 0.03, z0 + 0.5 * mesh['width'], f'{i + 1}#', va='center', fontsize=7)
            if np.isclose(T.min(), T_low):
                jz, jr = np.unravel_index(np.argmin(T), T.shape)
                ax.plot(mesh['r'][jr], mesh['z'][jz] + z0, marker='x', color='#00c8ff', markersize=5, markeredgewidth=1.2)
            z0 += mesh['width'] + gap
        ax.set_title(f'{tit}\n{run["time_h"][k]:.1f} h', fontsize=7)
        ax.set_xlabel('$r$ (m)')
        ax.set_xlim(0.2, 1.08)
        ax.set_xticks([0.3, 0.6, 0.9])
        ax.set_aspect('equal')
    axes[0].set_ylabel('$z$ (m)')
    fig.colorbar(cs, ax=axes, label='Temperatura (°C)', ticks=np.arange(30, 720, 90), shrink=0.85, aspect=25)
    save(fig, 'R6_campo_pila')


# ------------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------------

def write_csv(*, path: Path, header: list[str], rows: list[list]) -> None:
    '''
    Write rows to a csv file with 3 decimals for floats.

    Args:
        path:   output path
        header: column names
        rows:   table rows
    '''
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f'{v:.3f}' if isinstance(v, (float, np.floating)) else v for v in row])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--targets', type=float, nargs='+', default=DEFAULT_TARGETS_C, help='target temperatures in C')
    parser.add_argument('--hold-hours', type=float, default=60.0)
    parser.add_argument('--save-interval', type=float, default=300.0, help='s')
    args = parser.parse_args()
    targets = sorted(args.targets)
    if MAIN_TARGET_C not in targets:
        targets = sorted(targets + [MAIN_TARGET_C])
    OUT.mkdir(parents=True, exist_ok=True)

    runs = {g: run_case(grade=g, hold_hours=args.hold_hours, save_interval=args.save_interval) for g in GRADES}
    planar = run_case(grade='SAE1006', hold_hours=args.hold_hours, save_interval=args.save_interval, geometry='planar')

    times = {g: reach_times(run=runs[g], key='T_cold', targets_c=targets) for g in GRADES}
    probe_times = {g: reach_times(run=runs[g], key='probe', targets_c=targets) for g in GRADES}
    i_main = targets.index(MAIN_TARGET_C)
    times_main = {g: float(times[g][i_main]) for g in GRADES}
    k_main = nearest_index(run=runs['SAE1006'], t_h=times_main['SAE1006'])
    coil = int(runs['SAE1006']['cold_coil'][k_main])
    T_gas_c = runs['SAE1006']['case']['T_hold'] - TO_C
    heating_h = runs['SAE1006']['case']['heating_time'] / HOUR

    plot_geometry(run=runs['SAE1006'], t_main_h=times_main['SAE1006'])
    plot_properties()
    plot_coefficients(run=runs['SAE1006'])
    curv_rows = plot_curvature(axi=runs['SAE1006'], planar=planar, targets_c=targets)
    plot_curves(runs=runs, targets_c=targets, times=times, probe_times=probe_times)
    plot_times(targets_c=targets, times=times, probe_times=probe_times, T_gas_c=T_gas_c)
    plot_rz_maps(runs=runs, coil=coil, fixed_h=[3.0, 8.0, heating_h, 20.0, 30.0], times_main=times_main)
    plot_rt_maps(runs=runs, coil=coil, targets_c=targets, times_main=times_main)
    Q, stored, err = heat_by_surface(run=runs['SAE1006'], t_end_h=times_main['SAE1006'])
    Q10, _, err10 = heat_by_surface(run=runs['SAE1010'], t_end_h=times_main['SAE1010'])
    plot_heat_share(Q=Q)
    plot_stack_field(run=runs['SAE1006'], t_list_h=[heating_h, times_main['SAE1006']], titles=['Fin de la rampa', f'Punto frío a {MAIN_TARGET_C:.0f} °C'])

    rows = [[T, T + TO_C, a, b, (b - a) * 60.0, 100.0 * (b - a) / a] for T, a, b in zip(targets, times['SAE1006'], times['SAE1010'])]
    write_csv(path=OUT / 'tabla_tiempos.csv', header=['T_objetivo_C', 'T_objetivo_K', 't_SAE1006_h', 't_SAE1010_h', 'dt_min', 'dt_pct'], rows=rows)
    rows_p = [[T, a, b, (b - a) * 60.0] for T, a, b in zip(targets, probe_times['SAE1006'], probe_times['SAE1010'])]
    write_csv(path=OUT / 'tabla_referencia.csv', header=['T_objetivo_C', 't_SAE1006_h', 't_SAE1010_h', 'dt_min'], rows=rows_p)
    write_csv(path=OUT / 'tabla_curvatura.csv', header=['T_objetivo_C', 't_axisimetrico_h', 't_plano_h', 'dt_h', 'dt_pct'], rows=curv_rows)
    order = ['outer', 'inner', 'bottom', 'top']
    share = {g: 100.0 * q[:, [SURFACES.index(s) for s in order]] / q.sum(axis=1, keepdims=True) for g, q in (('SAE1006', Q), ('SAE1010', Q10))}
    write_csv(path=OUT / 'tabla_calor_superficies.csv', header=['grado', 'rollo', 'externa_pct', 'interna_pct', 'base_inf_pct', 'base_sup_pct', 'Q_total_MJ'], rows=[[g, f'{i + 1}#', *share[g][i], q.sum(axis=1)[i] / 1e6] for g, q in (('SAE1006', Q), ('SAE1010', Q10)) for i in range(len(q))])

    hf = runs['SAE1006']['h_function']
    lines = [
        f'Gas hold {T_gas_c:.2f} C, ramp {heating_h:.1f} h, hold {args.hold_hours:.0f} h, save interval {args.save_interval:.0f} s',
        f'Coil with the stack cold spot when it reaches {MAIN_TARGET_C:.0f} C: {coil + 1}#',
        f'Cold spot position at that time (SAE1006): r = {runs["SAE1006"]["meshes"][coil]["r"][np.unravel_index(np.argmin(runs["SAE1006"]["results"][coil]["T"][k_main]), runs["SAE1006"]["results"][coil]["T"][k_main].shape)[1]]:.3f} m',
        f'tau cold spot: SAE1006 {time_constant(run=runs["SAE1006"]):.2f} h, SAE1010 {time_constant(run=runs["SAE1010"]):.2f} h',
        f'max |T_cold 1006 - T_cold 1010| = {np.abs(runs["SAE1006"]["T_cold"] - runs["SAE1010"]["T_cold"]).max():.2f} K',
        'h_eff at the hold temperature, W/(m2 K), coil: outer inner bottom top',
    ]
    for i in range(len(runs['SAE1006']['meshes'])):
        h = hf(coil=i, T_g=T_gas_c + TO_C, time=0.0)
        lines.append(f'  {i + 1}#: {h["outer"]:.1f} {h["inner"]:.1f} {h["bottom"]:.1f} {h["top"]:.1f}')
    lines.append('Energy check, sum of surface heat / stored energy - 1, per coil (SAE1006, SAE1010):')
    lines.append('  ' + ' '.join(f'{e * 100:+.2f}%' for e in err) + ' | ' + ' '.join(f'{e * 100:+.2f}%' for e in err10))
    lines.append('Heat share per surface up to the main target (SAE1006), %: outer inner bottom top')
    for i in range(len(Q)):
        lines.append(f'  {i + 1}#: ' + ' '.join(f'{v:.1f}' for v in share['SAE1006'][i]))
    lines.append('Axisymmetric vs planar (SAE1006), target C: t_axi h, t_planar h, diff h, diff %')
    for row in curv_rows:
        lines.append('  ' + ', '.join(f'{v:.2f}' for v in row))
    lines.append('Cold spot times (target C: SAE1006 h, SAE1010 h, dt min)')
    for row in rows:
        lines.append(f'  {row[0]:.0f}: {row[2]:.3f}, {row[3]:.3f}, {row[4]:.1f}')
    lines.append('Reference point times (target C: SAE1006 h, SAE1010 h, dt min)')
    for row in rows_p:
        lines.append(f'  {row[0]:.0f}: {row[1]:.3f}, {row[2]:.3f}, {row[3]:.1f}')
    (OUT / 'resumen.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'Figures and tables in {OUT}')

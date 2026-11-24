'''
Assemble a full simulation case: coils, meshes, steel grades, furnace, contact and gas schedule.

Cases
    paper:   four-coil case of Yang et al. (2025), generic steel (EN 1993-1-2)
    ternium: Ternium data (furnace_parameters.TERNIUM), grades SAE 1006 and SAE 1010, mock stack
    load:    any real charge of Recocido -Tec.xlsx (3 to 5 coils), see build_load_case

Any other configuration (number of coils, sizes, grades, schedule) can be built with build_stack_case.
'''

import numpy as np
import pandas as pd

try:
    from .furnace_parameters import CONTACT, CONTACT_TERNIUM, FURNACE, FURNACE_TERNIUM, HOUR, LOAD_SCHEDULE, SCHEDULE_TERNIUM, TABLE_3, TERNIUM, TERNIUM_COILS
    from .mesh import build_stack, coil_inner_diameter
    from .properties import STEEL_GRADES, steel_density
    from .solver import trapezoid_schedule
except ImportError:
    from furnace_parameters import CONTACT, CONTACT_TERNIUM, FURNACE, FURNACE_TERNIUM, HOUR, LOAD_SCHEDULE, SCHEDULE_TERNIUM, TABLE_3, TERNIUM, TERNIUM_COILS
    from mesh import build_stack, coil_inner_diameter
    from properties import STEEL_GRADES, steel_density
    from solver import trapezoid_schedule

CASES = ('paper', 'ternium')


def stack_height(*, widths: list[float], plate_gap: float, top_gap: float) -> float:
    '''
    Compute the height taken by a coil stack under the inner cover.

    Coils plus one convector plate between each pair plus the space above the top coil. The base
    convector is not counted (no data).

    Args:
        widths:    coil widths in m, bottom coil first
        plate_gap: height of a convector plate in m
        top_gap:   height of the space above the top coil in m

    Returns:
        Stack height in m.
    '''
    return float(sum(widths) + (len(widths) - 1) * plate_gap + top_gap)


def check_stack(*, coils: list[dict], furnace: dict, strict: bool = True, rtol: float = 2e-3) -> list[str]:
    '''
    Check a stack against the coil limits and the bell size given by Ternium.

    Real charges sit on the limits (thickness 3.491 mm against 3.49 mm, outer diameters from the mass
    slightly above 1.829 m), so limits are compared with a relative tolerance.

    Args:
        coils:   coil dicts with keys inner_diameter, outer_diameter, width, thickness, grade
        furnace: furnace parameters (plate_gap, top_gap)
        strict:  true raises on the first problem, false only returns the list
        rtol:    relative tolerance on the Ternium limits

    Returns:
        List of problems found, empty if the stack is valid.

    Raises:
        ValueError if strict and a coil is outside the Ternium limits, a grade is unknown, the
        coils do not fit the bell diameter or the stack is taller than the operating height.
    '''
    t = TERNIUM
    problems = []
    if not 2 <= len(coils) <= 6:
        problems.append(f'stack must have 2 to 6 coils, got {len(coils)}')
    for i, c in enumerate(coils, start=1):
        if not np.isclose(c['inner_diameter'], t['inner_diameter']):
            problems.append(f"coil {i}#: inner diameter must be {t['inner_diameter']} m")
        if not c['inner_diameter'] < c['outer_diameter'] <= t['outer_diameter_max'] * (1.0 + rtol):
            problems.append(f"coil {i}#: outer diameter {c['outer_diameter']:.3f} m must be in ({t['inner_diameter']}, {t['outer_diameter_max']}] m")
        if not t['width_min'] * (1.0 - rtol) <= c['width'] <= t['width_max'] * (1.0 + rtol):
            problems.append(f"coil {i}#: width {c['width']:.3f} m must be in [{t['width_min']}, {t['width_max']}] m")
        if not t['thickness_min'] * (1.0 - rtol) <= c['thickness'] <= t['thickness_max'] * (1.0 + rtol):
            problems.append(f"coil {i}#: thickness {c['thickness'] * 1e3:.3f} mm must be in [{t['thickness_min'] * 1e3}, {t['thickness_max'] * 1e3}] mm")
        if c['grade'] not in STEEL_GRADES:
            problems.append(f"coil {i}#: unknown grade {c['grade']!r}, use one of {sorted(STEEL_GRADES)}")
        if c['outer_diameter'] >= t['bell_diameter']:
            problems.append(f'coil {i}#: outer diameter must be smaller than the bell diameter')

    height = stack_height(widths=[c['width'] for c in coils], plate_gap=furnace['plate_gap'], top_gap=furnace['top_gap'])
    if height > t['operating_height']:
        problems.append(f"stack height {height:.3f} m is above the operating height {t['operating_height']} m")

    if strict and problems:
        raise ValueError('; '.join(problems))
    return problems


def check_ternium_coils(*, coils: list[dict], furnace: dict) -> None:
    '''
    Check the mock Ternium stack, kept for the existing scripts. See check_stack.

    Args:
        coils:   coil dicts with keys inner_diameter, outer_diameter, width, thickness, grade
        furnace: furnace parameters (plate_gap, top_gap)

    Raises:
        ValueError if the stack does not pass check_stack.
    '''
    check_stack(coils=coils, furnace=furnace, strict=True, rtol=0.0)


def coil_mass(*, coil: dict) -> float:
    '''
    Compute the mass of a coil as a solid steel ring.

    Args:
        coil:   coil dict with keys inner_diameter, outer_diameter, width and grade

    Returns:
        Mass in kg.
    '''
    ring = np.pi / 4.0 * (coil['outer_diameter']**2 - coil['inner_diameter']**2)
    return float(steel_density(coil['grade']) * ring * coil['width'])


def build_stack_case(*, coils: list[dict], schedule: dict, furnace: dict = FURNACE_TERNIUM, contact: dict = CONTACT_TERNIUM, name: str = 'custom', reference: dict | None = None, n_r: int = 21, n_z: int = 31, strict: bool = True) -> dict:
    '''
    Build every input of a simulation for any stack and gas schedule.

    This is the general entry point: the number of coils, their sizes, grades and the schedule are
    all free. build_case and build_load_case call it.

    Args:
        coils:     coil dicts, bottom coil first, with keys inner_diameter, outer_diameter, width,
                   thickness (m) and grade; mass (kg) is computed if missing
        schedule:  trapezoid_schedule arguments: T_start, T_hold, T_end (K), heating_time,
                   holding_time, cooling_time (s)
        furnace:   furnace parameters (see furnace_parameters.FURNACE_TERNIUM)
        contact:   contact parameters (see furnace_parameters.CONTACT_TERNIUM)
        name:      case name, used for the figure folder
        reference: None, or dict with coil (index), r (m) and T (K) of a reference point
        n_r:       radial nodes per coil
        n_z:       axial nodes per coil
        strict:    passed to check_stack; false keeps going and stores the problems in the case

    Returns:
        Case dict, see build_case, plus the key problems (list of str from check_stack).

    Raises:
        ValueError if strict and the stack does not pass check_stack, or the reference point is
        outside its coil.
    '''
    coils = [dict(c) for c in coils]
    for c in coils:
        if 'mass' not in c:
            c['mass'] = coil_mass(coil=c)
    problems = check_stack(coils=coils, furnace=furnace, strict=strict)

    schedule_time, schedule_temperature = trapezoid_schedule(**schedule)
    meshes = build_stack(coils=coils, n_r=n_r, n_z=n_z)
    if reference is not None and not meshes[reference['coil']]['r_in'] <= reference['r'] <= meshes[reference['coil']]['r_out']:
        raise ValueError('the reference point is outside the coil, check the outer diameter')

    return {
        'name': name,
        'coils': coils,
        'meshes': meshes,
        'grades': [c['grade'] for c in coils],
        'strip_thickness': [c['thickness'] for c in coils],
        'furnace': furnace,
        'contact': contact,
        'schedule': dict(schedule),
        'schedule_time': schedule_time,
        'schedule_temperature': schedule_temperature,
        'T_initial': schedule['T_start'],
        'T_hold': schedule['T_hold'],
        'heating_time': schedule['heating_time'],
        'holding_time': schedule['holding_time'],
        'reference': reference,
        'problems': problems,
    }


def load_schedule(*, T_sat: float, t_sat: float, heating_ramp: float = LOAD_SCHEDULE['heating_ramp'], hold_horizon: float = LOAD_SCHEDULE['hold_horizon'], T_start: float = LOAD_SCHEDULE['T_start'], T_end: float = LOAD_SCHEDULE['T_end'], cooling_time: float = LOAD_SCHEDULE['cooling_time']) -> dict:
    '''
    Build the gas schedule of a real charge from its saturation temperature and time.

    The data gives the saturation (soak) temperature and time, not the heating curve. The gas rises
    linearly to T_sat in heating_ramp and is held long enough for the coldest point to get there
    (hold_horizon) plus the soak time of the data, then cools. The heating time of the charge is an
    output of the model (see metrics.stack_metrics), not an input.

    Args:
        T_sat:        saturation temperature in K (RH- Temperatura Saturacion)
        t_sat:        saturation time in s (RH- Tiempo Saturacion)
        heating_ramp: time for the gas to reach T_sat in s
        hold_horizon: extra hold time before the soak, so the cold spot can reach T_sat, in s
        T_start:      initial charge and gas temperature in K
        T_end:        gas temperature at the end of cooling in K
        cooling_time: duration of the cooling ramp in s

    Returns:
        Dict with the arguments of solver.trapezoid_schedule.

    Raises:
        ValueError if T_sat is not above T_start or a time is negative.
    '''
    if not T_sat > T_start:
        raise ValueError('T_sat must be above T_start')
    if min(t_sat, heating_ramp, hold_horizon, cooling_time) < 0.0:
        raise ValueError('times must be non-negative')
    return {'T_start': T_start, 'T_hold': T_sat, 'T_end': T_end, 'heating_time': heating_ramp, 'holding_time': hold_horizon + t_sat, 'cooling_time': cooling_time}


def build_case(*, case: str = 'ternium', grade: str | None = None, n_r: int = 21, n_z: int = 31) -> dict:
    '''
    Build every input of a simulation.

    Args:
        case:  'paper' or 'ternium'
        grade: steel grade forced on every coil (key of properties.STEEL_GRADES); None keeps the
               grade of each coil in the case table
        n_r:   radial nodes per coil
        n_z:   axial nodes per coil

    Returns:
        Dict with keys:
            name:                 case name, with the forced grade if any
            coils:                coil dicts with inner_diameter, outer_diameter, width, thickness,
                                  grade and mass (kg)
            meshes:               mesh dicts from mesh.build_stack, bottom coil first
            grades:               steel grade of each coil
            strip_thickness:      strip thickness of each coil in m
            furnace:              furnace parameters
            contact:              contact parameters
            schedule_time:        gas schedule times in s
            schedule_temperature: gas schedule temperatures in K
            T_initial:            initial coil temperature in K
            T_hold:               holding gas temperature in K
            heating_time:         end of the heating ramp in s
            holding_time:         duration of the hold in s
            reference:            None, or dict with coil, r (m) and T (K) of the Ternium reference point
            schedule:             arguments of the gas schedule
            problems:             problems found by check_stack (empty for valid stacks)

    Raises:
        ValueError if case or grade is unknown, or a Ternium coil is outside the Ternium limits.
    '''
    if case not in CASES:
        raise ValueError(f'case must be one of {CASES}')
    if grade is not None and grade not in STEEL_GRADES:
        raise ValueError(f'unknown steel grade {grade!r}, use one of {sorted(STEEL_GRADES)}')

    name = case if grade is None else f'{case}_{grade}'
    if case == 'paper':
        coils = []
        for c in TABLE_3:
            g = grade or 'generic'
            inner_diameter = coil_inner_diameter(mass=c['mass'], outer_diameter=c['outer_diameter'], width=c['width'], density=steel_density(g))
            coils.append({'inner_diameter': inner_diameter, 'outer_diameter': c['outer_diameter'], 'width': c['width'], 'thickness': c['thickness'], 'grade': g, 'mass': c['mass']})
        schedule = {'T_start': 303.15, 'T_hold': 973.15, 'T_end': 423.15, 'heating_time': 12.0 * HOUR, 'holding_time': 10.0 * HOUR, 'cooling_time': 20.0 * HOUR}
    else:
        coils = []
        for c in TERNIUM_COILS:
            coil = dict(c)
            if grade is not None:
                coil['grade'] = grade
            coil['mass'] = coil_mass(coil=coil)
            coils.append(coil)
        check_ternium_coils(coils=coils, furnace=FURNACE_TERNIUM)
        k = TERNIUM['probe_coil']
        reference = {'coil': k, 'r': 0.5 * coils[k]['inner_diameter'] + TERNIUM['probe_offset'], 'T': TERNIUM['T_contact']}
        return build_stack_case(coils=coils, schedule=SCHEDULE_TERNIUM, furnace=FURNACE_TERNIUM, contact=CONTACT_TERNIUM, name=name, reference=reference, n_r=n_r, n_z=n_z)

    return build_stack_case(coils=coils, schedule=schedule, furnace=FURNACE, contact=CONTACT, name=name, reference=None, n_r=n_r, n_z=n_z, strict=False)


def build_load_case(*, load_id: int, coils_table: pd.DataFrame, loads_table: pd.DataFrame, T_sat_default: float | None = None, n_r: int = 21, n_z: int = 31, **schedule_kwargs: float) -> dict:
    '''
    Build the case of one real charge of Recocido -Tec.xlsx.

    Args:
        load_id:        value of RH- Carga
        coils_table:    coils data frame from pipeline.data_processing.load_data
        loads_table:    loads data frame from pipeline.data_processing.load_data
        T_sat_default:  saturation temperature in K used when the charge has none (35 % of the
                        charges have 0 F); None raises instead
        n_r:            radial nodes per coil
        n_z:            axial nodes per coil
        schedule_kwargs: optional heating_ramp, hold_horizon, T_start, T_end, cooling_time for
                        load_schedule

    Returns:
        Case dict (see build_stack_case) plus load_id, load (row of loads_table as dict) and
        T_sat_imputed (true when T_sat_default was used).

    Raises:
        KeyError if the charge is not in the tables; ValueError if T_sat is missing and
        T_sat_default is None.
    '''
    load = loads_table.loc[load_id]
    T_sat = load['T_sat']
    imputed = bool(np.isnan(T_sat))
    if imputed:
        if T_sat_default is None:
            raise ValueError(f'load {load_id} has no saturation temperature, pass T_sat_default')
        T_sat = T_sat_default
    t_sat = max(float(load['t_sat']), 0.0)

    rows = coils_table[coils_table['load_id'] == load_id].sort_values('position')
    if rows.empty:
        raise KeyError(f'load {load_id} has no coils')
    keys = ['coil_id', 'inner_diameter', 'outer_diameter', 'width', 'thickness', 'grade', 'family', 'mass']
    coils = [{k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()} for row in rows[keys].to_dict('records')]
    schedule = load_schedule(T_sat=float(T_sat), t_sat=t_sat, **schedule_kwargs)
    case = build_stack_case(coils=coils, schedule=schedule, furnace=FURNACE_TERNIUM, contact=CONTACT_TERNIUM, name=f'load_{load_id}', n_r=n_r, n_z=n_z, strict=False)
    case.update({'load_id': load_id, 'load': load.to_dict(), 'T_sat_imputed': imputed})
    return case


def probe_temperature(*, mesh: dict, result: dict, r: float) -> np.ndarray:
    '''
    Interpolate the coil temperature at a radius, for every saved time and every axial node.

    Args:
        mesh:   mesh dict of the coil
        result: result dict of the coil from solver.simulate_coil
        r:      radius in m, between r_in and r_out

    Returns:
        Temperatures in K, shape (n_save, n_z).
    '''
    T = result['T']
    j = int(np.clip(np.searchsorted(mesh['r'], r) - 1, 0, mesh['n_r'] - 2))
    w = (r - mesh['r'][j]) / (mesh['r'][j + 1] - mesh['r'][j])
    return (1.0 - w) * T[:, :, j] + w * T[:, :, j + 1]

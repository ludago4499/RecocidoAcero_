'''
Assemble a full simulation case: coils, meshes, steel grades, furnace, contact and gas schedule.

Cases
    paper:   four-coil case of Yang et al. (2025), generic steel (EN 1993-1-2)
    ternium: Ternium data (furnace_parameters.TERNIUM), grades SAE 1006 and SAE 1010
'''

import numpy as np

try:
    from .furnace_parameters import CONTACT, CONTACT_TERNIUM, FURNACE, FURNACE_TERNIUM, HOUR, SCHEDULE_TERNIUM, TABLE_3, TERNIUM, TERNIUM_COILS
    from .mesh import build_stack, coil_inner_diameter
    from .properties import STEEL_GRADES, steel_density
    from .solver import trapezoid_schedule
except ImportError:
    from furnace_parameters import CONTACT, CONTACT_TERNIUM, FURNACE, FURNACE_TERNIUM, HOUR, SCHEDULE_TERNIUM, TABLE_3, TERNIUM, TERNIUM_COILS
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


def check_ternium_coils(*, coils: list[dict], furnace: dict) -> None:
    '''
    Check a Ternium stack against the coil limits and the bell size given by Ternium.

    Args:
        coils:   coil dicts with keys inner_diameter, outer_diameter, width, thickness, grade
        furnace: furnace parameters (plate_gap, top_gap)

    Raises:
        ValueError if a coil is outside the Ternium limits, a grade is not a Ternium grade, the
        coils do not fit the bell diameter or the stack is taller than the operating height.
    '''
    t = TERNIUM
    for i, c in enumerate(coils, start=1):
        if not np.isclose(c['inner_diameter'], t['inner_diameter']):
            raise ValueError(f"coil {i}#: inner diameter must be {t['inner_diameter']} m")
        if not c['inner_diameter'] < c['outer_diameter'] <= t['outer_diameter_max']:
            raise ValueError(f"coil {i}#: outer diameter must be in ({t['inner_diameter']}, {t['outer_diameter_max']}] m")
        if not t['width_min'] <= c['width'] <= t['width_max']:
            raise ValueError(f"coil {i}#: width must be in [{t['width_min']}, {t['width_max']}] m")
        if not t['thickness_min'] <= c['thickness'] <= t['thickness_max']:
            raise ValueError(f"coil {i}#: thickness must be in [{t['thickness_min']}, {t['thickness_max']}] m")
        if c['grade'] not in t['grades']:
            raise ValueError(f"coil {i}#: grade must be one of {t['grades']}")
        if c['outer_diameter'] >= t['bell_diameter']:
            raise ValueError(f'coil {i}#: outer diameter must be smaller than the bell diameter')

    height = stack_height(widths=[c['width'] for c in coils], plate_gap=furnace['plate_gap'], top_gap=furnace['top_gap'])
    if height > t['operating_height']:
        raise ValueError(f"stack height {height:.3f} m is above the operating height {t['operating_height']} m")


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

    Raises:
        ValueError if case or grade is unknown, or a Ternium coil is outside the Ternium limits.
    '''
    if case not in CASES:
        raise ValueError(f'case must be one of {CASES}')
    if grade is not None and grade not in STEEL_GRADES:
        raise ValueError(f'unknown steel grade {grade!r}, use one of {sorted(STEEL_GRADES)}')

    if case == 'paper':
        coils = []
        for c in TABLE_3:
            g = grade or 'generic'
            inner_diameter = coil_inner_diameter(mass=c['mass'], outer_diameter=c['outer_diameter'], width=c['width'], density=steel_density(g))
            coils.append({'inner_diameter': inner_diameter, 'outer_diameter': c['outer_diameter'], 'width': c['width'], 'thickness': c['thickness'], 'grade': g, 'mass': c['mass']})
        furnace, contact = FURNACE, CONTACT
        schedule = {'T_start': 303.15, 'T_hold': 973.15, 'T_end': 423.15, 'heating_time': 12.0 * HOUR, 'holding_time': 10.0 * HOUR, 'cooling_time': 20.0 * HOUR}
        reference = None
    else:
        coils = []
        for c in TERNIUM_COILS:
            coil = dict(c)
            if grade is not None:
                coil['grade'] = grade
            ring = np.pi / 4.0 * (coil['outer_diameter']**2 - coil['inner_diameter']**2)
            coil['mass'] = steel_density(coil['grade']) * ring * coil['width']
            coils.append(coil)
        furnace, contact = FURNACE_TERNIUM, CONTACT_TERNIUM
        check_ternium_coils(coils=coils, furnace=furnace)
        schedule = SCHEDULE_TERNIUM
        k = TERNIUM['probe_coil']
        reference = {'coil': k, 'r': 0.5 * coils[k]['inner_diameter'] + TERNIUM['probe_offset'], 'T': TERNIUM['T_contact']}

    schedule_time, schedule_temperature = trapezoid_schedule(**schedule)
    meshes = build_stack(coils=coils, n_r=n_r, n_z=n_z)
    if reference is not None and not meshes[reference['coil']]['r_in'] <= reference['r'] <= meshes[reference['coil']]['r_out']:
        raise ValueError('the Ternium reference point is outside the coil, check the outer diameter')

    return {
        'name': case if grade is None else f'{case}_{grade}',
        'coils': coils,
        'meshes': meshes,
        'grades': [c['grade'] for c in coils],
        'strip_thickness': [c['thickness'] for c in coils],
        'furnace': furnace,
        'contact': contact,
        'schedule_time': schedule_time,
        'schedule_temperature': schedule_temperature,
        'T_initial': schedule['T_start'],
        'T_hold': schedule['T_hold'],
        'heating_time': schedule['heating_time'],
        'holding_time': schedule['holding_time'],
        'reference': reference,
    }


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

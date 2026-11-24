'''
Physical, geometric and operational constraints of a charge, shared by the optimizer and the physical model.

A candidate charge (the x of the optimizer) is a dict:
    coils:     list of coil dicts, bottom coil first, with inner_diameter, outer_diameter, width,
               thickness (m), grade and mass (kg); the coils of the Excel come from
               data_processing.build_coils
    T_sat:     saturation temperature in K
    t_sat:     saturation (soak) time in s
    furnace_id, base_id, cooler_id: optional equipment ids

Geometric limits (coil sizes, bell diameter, stack height) live in bell_furnace (cases.check_stack and
furnace_parameters.TERNIUM) so the physical model uses the same ones. This module adds the operational
ranges seen in the plant data and the thermal condition from the physical model.

Owner: shared (main). Changes here affect every branch.
'''

import pandas as pd

from bell_furnace.cases import check_stack
from bell_furnace.furnace_parameters import COLD_SPOT_MARGIN, FURNACE_TERNIUM, fahrenheit_to_kelvin

HOUR = 3600.0

# Operational ranges of the valid charges in Recocido -Tec.xlsx (520 charges). Recompute with
# bounds_from_data when the data changes. These are what the plant has done, not hard limits.
BOUNDS = {
    'n_coils': (3, 5),
    'load_mass': (45.4e3, 87.0e3),                                       # kg
    'T_sat': (fahrenheit_to_kelvin(1000.0), fahrenheit_to_kelvin(1298.0)),  # K
    't_sat': (0.0, 20.0 * HOUR),                                         # s
}


def bounds_from_data(*, loads: pd.DataFrame) -> dict[str, tuple[float, float]]:
    '''
    Compute the operational ranges from the valid charges of the data.

    Args:
        loads:  data frame from data_processing.build_loads

    Returns:
        Dict with the keys of BOUNDS and (min, max) of the valid charges.
    '''
    valid = loads[loads['valid']]
    return {
        'n_coils': (int(valid['n_coils'].min()), int(valid['n_coils'].max())),
        'load_mass': (float(valid['load_mass'].min()), float(valid['load_mass'].max())),
        'T_sat': (float(valid['T_sat'].min()), float(valid['T_sat'].max())),
        't_sat': (float(valid['t_sat'].min()), float(valid['t_sat'].max())),
    }


def check_load(*, load: dict, bounds: dict[str, tuple[float, float]] = BOUNDS, furnace: dict = FURNACE_TERNIUM) -> list[str]:
    '''
    Check a candidate charge against the geometric limits and the operational ranges.

    Args:
        load:    candidate charge, see the module docstring
        bounds:  operational ranges, see BOUNDS
        furnace: furnace parameters with plate_gap and top_gap, for the stack height

    Returns:
        List of violated constraints, empty if the charge is feasible.
    '''
    problems = check_stack(coils=load['coils'], furnace=furnace, strict=False)
    values = {
        'n_coils': len(load['coils']),
        'load_mass': float(sum(c['mass'] for c in load['coils'])),
        'T_sat': load['T_sat'],
        't_sat': load['t_sat'],
    }
    for key, value in values.items():
        low, high = bounds[key]
        if not low <= value <= high:
            problems.append(f'{key} = {value:.4g} outside [{low:.4g}, {high:.4g}]')
    return problems


def thermal_constraint(*, metrics: dict, max_dT: float | None = None) -> list[str]:
    '''
    Check the thermal condition from a physical model run: the cold spot must reach T_sat - margin.

    Minimizing time alone does not guarantee a correct annealing, so every candidate the optimizer
    keeps must pass this check with the physical model.

    Args:
        metrics: dict from bell_furnace.metrics.stack_metrics
        max_dT:  optional largest hot to cold spot difference allowed in K; None skips it

    Returns:
        List of violated thermal constraints, empty if the charge anneals.
    '''
    problems = []
    if not metrics['reached']:
        problems.append(f'cold spot does not reach T_sat - {COLD_SPOT_MARGIN:g} K')
    if max_dT is not None and metrics['dT_max'] > max_dT:
        problems.append(f"hot to cold spot difference {metrics['dT_max']:.0f} K above {max_dT:g} K")
    return problems


def penalty(*, problems: list[str], weight: float = 1.0e3) -> float:
    '''
    Convert a list of violated constraints into a penalty for the objective function.

    Args:
        problems: list from check_load or thermal_constraint
        weight:   penalty per violated constraint, in the units of the objective

    Returns:
        weight times the number of violated constraints.
    '''
    return float(weight * len(problems)) if problems else 0.0


if __name__ == '__main__':
    from pipeline.data_processing import load_data

    coils, loads = load_data()
    print('bounds from data:', bounds_from_data(loads=loads))
    load_id = loads[loads['valid']].index[0]
    rows = coils[coils['load_id'] == load_id].sort_values('position')
    load = {'coils': rows.to_dict('records'), 'T_sat': loads.loc[load_id, 'T_sat'], 't_sat': loads.loc[load_id, 't_sat']}
    print(f'charge {load_id}:', check_load(load=load) or 'feasible')

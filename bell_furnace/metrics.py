'''
Run a case and reduce the temperature fields to the numbers that can be compared with the plant data.

Model outputs per charge
    t_heat_cold:  time for the coldest point of the whole stack to reach T_sat - margin (heating time)
    t_heat_mean:  time for the mass-weighted mean temperature of the stack to reach T_sat - margin
    slowest_coil: coil (1# is the bottom) that holds the cold spot at t_heat_cold
    dT_max:       largest hot to cold spot difference in the stack during heating
    energy:       heat absorbed by the steel from T_start to T_sat
    t_model:      t_heat_cold + soak time of the data, the model estimate of heating plus soak

The plant data gives the net cycle time (heating, soak and cooling) and the soak time, so the
comparable data value is t_net - t_sat (heating plus cooling). Cooling is not modelled (separate
cooler hood, no data on its gas), so the two numbers are compared by trend, not one to one.
'''

import numpy as np

try:
    from .furnace_parameters import COLD_SPOT_MARGIN
    from .properties import steel_properties
    from .solver import make_h_function, make_lambda_r_function, simulate_stack, time_to_reach
except ImportError:
    from furnace_parameters import COLD_SPOT_MARGIN
    from properties import steel_properties
    from solver import make_h_function, make_lambda_r_function, simulate_stack, time_to_reach

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # numpy 2 renamed trapz


def heating_schedule(*, case: dict) -> tuple[np.ndarray, np.ndarray]:
    '''
    Cut the gas schedule of a case at the end of the hold, before the soak and cooling.

    Enough to get every heating metric and about half the run time of the full schedule.

    Args:
        case:   case dict from cases.build_stack_case (needs the key schedule)

    Returns:
        Tuple (time in s, gas temperature in K) for solver.simulate_stack.
    '''
    s = case['schedule']
    t_ramp = s['heating_time']
    t_hold_end = t_ramp + s['holding_time']
    return np.array([0.0, t_ramp, t_hold_end]), np.array([s['T_start'], s['T_hold'], s['T_hold']])


def run_case(*, case: dict, heating_only: bool = False, dt_max: float = 30.0, save_interval: float = 600.0) -> list[dict]:
    '''
    Simulate every coil of a case.

    Args:
        case:          case dict from cases.build_stack_case, build_case or build_load_case
        heating_only:  true stops at the end of the hold (see heating_schedule)
        dt_max:        largest time step in s
        save_interval: time between saved results in s

    Returns:
        List of result dicts from solver.simulate_stack, bottom coil first.
    '''
    if heating_only:
        schedule_time, schedule_temperature = heating_schedule(case=case)
    else:
        schedule_time, schedule_temperature = case['schedule_time'], case['schedule_temperature']
    h_function = make_h_function(meshes=case['meshes'], furnace=case['furnace'])
    lambda_r_function = make_lambda_r_function(strip_thickness=case['strip_thickness'], contact=case['contact'], grades=case['grades'])
    return simulate_stack(meshes=case['meshes'], T_initial=case['T_initial'], schedule_time=schedule_time, schedule_temperature=schedule_temperature, h_function=h_function, lambda_r_function=lambda_r_function, dt_max=dt_max, save_interval=save_interval, grades=case['grades'])


def absorbed_energy(*, coils: list[dict], T_start: float, T_end: float, n: int = 200) -> float:
    '''
    Compute the heat needed to bring every coil from T_start to T_end, from the cp table of its grade.

    Args:
        coils:   coil dicts with keys mass (kg) and grade
        T_start: initial temperature in K
        T_end:   final temperature in K
        n:       points of the cp integration

    Returns:
        Energy in J.
    '''
    T = np.linspace(T_start, T_end, n)
    total = 0.0
    for c in coils:
        _, cp, _ = steel_properties(T, grade=c['grade'])
        total += c['mass'] * _trapezoid(cp, T)
    return float(total)


def stack_metrics(*, case: dict, results: list[dict], margin: float = COLD_SPOT_MARGIN) -> dict:
    '''
    Reduce the results of a stack to the heating metrics described in the module docstring.

    Args:
        case:    case dict used for the run
        results: result dicts from run_case, bottom coil first
        margin:  the stack is heated when its temperature is within this of T_sat, in K

    Returns:
        Dict with keys t_heat_cold, t_heat_mean (s, nan if not reached inside the run),
        slowest_coil (1 is the bottom, 0 if not reached), dT_max (K), energy (J), t_model (s, nan
        if t_sat is not known), t_heat_coils (list, s, per coil) and reached (bool).
    '''
    time = results[0]['time']
    target = case['T_hold'] - margin
    T_cold = np.min([r['T_min'] for r in results], axis=0)
    T_hot = np.max([r['T_max'] for r in results], axis=0)
    masses = np.array([c['mass'] for c in case['coils']])
    T_mean = np.sum([m * r['T_mean'] for m, r in zip(masses, results)], axis=0) / masses.sum()

    t_heat_cold = time_to_reach(time=time, values=T_cold, target=target)
    t_heat_coils = [time_to_reach(time=time, values=r['T_min'], target=target) for r in results]
    reached = bool(np.isfinite(t_heat_cold))
    slowest = int(np.nanargmax(t_heat_coils)) + 1 if reached else 0

    heating = time <= (t_heat_cold if reached else time[-1])
    t_sat = case.get('load', {}).get('t_sat', np.nan)

    return {
        't_heat_cold': t_heat_cold,
        't_heat_mean': time_to_reach(time=time, values=T_mean, target=target),
        'slowest_coil': slowest,
        'dT_max': float(np.max(T_hot[heating] - T_cold[heating])),
        'energy': absorbed_energy(coils=case['coils'], T_start=case['T_initial'], T_end=case['T_hold']),
        't_model': t_heat_cold + max(float(t_sat), 0.0) if np.isfinite(t_sat) else float('nan'),
        't_heat_coils': t_heat_coils,
        'reached': reached,
    }

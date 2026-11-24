'''
Tests for the real-charge pipeline: data_processing, cases.build_stack_case / build_load_case and metrics.

The tests that read Recocido -Tec.xlsx are skipped when the file is not in the project root.

Run from the project root with: python -m pytest tests
'''

import numpy as np
import pytest

from bell_furnace.cases import build_case, build_load_case, build_stack_case, check_stack, coil_mass, load_schedule
from bell_furnace.furnace_parameters import FURNACE_TERNIUM, HOUR
from pipeline.constraints import check_load, penalty
from pipeline.data_processing import EXCEL_PATH, outer_diameter_from_mass

needs_excel = pytest.mark.skipif(not EXCEL_PATH.exists(), reason='Recocido -Tec.xlsx not in the project root')


def five_coils() -> list[dict]:
    '''
    Build a valid five-coil stack inside the Ternium limits.

    Returns:
        List of five coil dicts, bottom coil first.
    '''
    return [{'inner_diameter': 0.508, 'outer_diameter': 1.70, 'width': 0.90, 'thickness': 1.0e-3, 'grade': g} for g in ('SAE1006', 'SAE1006', 'IF', 'SAE1010', 'SAE1006')]


def test_outer_diameter_round_trip() -> None:
    '''
    The outer diameter from the mass gives back the same mass as a solid ring.
    '''
    coil = {'inner_diameter': 0.508, 'width': 1.2, 'grade': 'SAE1006', 'outer_diameter': 1.75}
    mass = coil_mass(coil=coil)
    D = outer_diameter_from_mass(mass=np.array([mass]), width=np.array([1.2]), inner_diameter=0.508, density=np.array([7850.0]))
    assert D[0] == pytest.approx(1.75, rel=1e-3)


def test_check_stack_tolerance() -> None:
    '''
    A thickness 0.001 mm above the Ternium maximum passes with the default tolerance, not with zero.
    '''
    coils = five_coils()
    coils[0]['thickness'] = 3.491e-3
    assert check_stack(coils=coils, furnace=FURNACE_TERNIUM, strict=False) == []
    assert check_stack(coils=coils, furnace=FURNACE_TERNIUM, strict=False, rtol=0.0)


def test_five_coil_case() -> None:
    '''
    A five-coil stack with three grades builds one mesh, mass and grade per coil.
    '''
    schedule = load_schedule(T_sat=940.0, t_sat=8.0 * HOUR)
    case = build_stack_case(coils=five_coils(), schedule=schedule, n_r=7, n_z=7)
    assert len(case['meshes']) == 5
    assert case['grades'] == ['SAE1006', 'SAE1006', 'IF', 'SAE1010', 'SAE1006']
    assert all(c['mass'] > 0.0 for c in case['coils'])
    assert case['T_hold'] == 940.0


def test_existing_cases_unchanged() -> None:
    '''
    The paper and mock Ternium cases still build with four coils.
    '''
    assert len(build_case(case='paper')['meshes']) == 4
    assert build_case(case='ternium')['problems'] == []


@needs_excel
def test_real_load_heating() -> None:
    '''
    A real charge builds with its coils ordered bottom to top and heats inside the horizon.
    '''
    from bell_furnace.metrics import run_case, stack_metrics
    from pipeline.data_processing import load_data

    coils, loads = load_data()
    load_id = loads[loads['valid']].index[0]
    case = build_load_case(load_id=load_id, coils_table=coils, loads_table=loads, n_r=7, n_z=7)
    assert len(case['coils']) == loads.loc[load_id, 'n_coils']
    metrics = stack_metrics(case=case, results=run_case(case=case, heating_only=True, dt_max=300.0))
    assert metrics['reached']
    assert 0.0 < metrics['t_heat_mean'] <= metrics['t_heat_cold']


def test_check_load_bounds() -> None:
    '''
    A five-coil candidate inside every range is feasible; a soak time out of range is reported.
    '''
    coils = [dict(c, mass=coil_mass(coil=c)) for c in five_coils()]
    load = {'coils': coils, 'T_sat': 940.0, 't_sat': 8.0 * HOUR}
    assert check_load(load=load) == []
    load['t_sat'] = 30.0 * HOUR
    problems = check_load(load=load)
    assert len(problems) == 1 and penalty(problems=problems) > 0.0


@needs_excel
def test_features_match_team_cleaning() -> None:
    '''
    load_features keeps the same 520 charges as analisis_fase2.py, and one column per position.
    '''
    from pipeline.data_processing import load_data, load_features, position_features

    coils, loads = load_data()
    features = load_features(loads=loads)
    assert len(features) == 520
    assert {'Peso_Carga_t', 'Temp_Sat_C', 'Tiempo_Neto_h', 'lab_elongation'} <= set(features.columns)
    assert position_features(coils=coils).shape[1] == 15

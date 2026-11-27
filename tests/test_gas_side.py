'''
Tests for the Option C gas side: flow distribution, scaled channels and the surface model.

Run from the project root with: python -m pytest tests
'''

import numpy as np
import pytest

from bell_furnace.convection import laminar_floor, scaled_channel_geometry
from bell_furnace.flow import surface_flows
from bell_furnace.furnace_parameters import FURNACE_TERNIUM
from bell_furnace.mesh import build_stack
from bell_furnace.solver import flow_report, make_surface_model, surface_group

ZETA = 1.0 / 0.909


def stack(n: int = 4) -> list[dict]:
    '''
    Build the meshes of a stack of equal coils.

    Args:
        n: number of coils

    Returns:
        Mesh dicts, bottom coil first.
    '''
    coils = [{'inner_diameter': 0.508, 'outer_diameter': 1.75, 'width': 1.1, 'thickness': 1.2e-3, 'grade': 'SAE1006'} for _ in range(n)]
    return build_stack(coils=coils, n_r=5, n_z=5)


def scaled(**changes: object) -> dict:
    '''
    Return the Ternium furnace with the scaled channel model, Yang bottom, rated flow and the given changes.

    Args:
        changes: furnace keys to override

    Returns:
        Furnace dict.
    '''
    return {**FURNACE_TERNIUM, 'channel_model': 'scaled', 'bottom_mode': 'yang', 'flow_scale': 1.0, 'psi_groups': {}, **changes}


def test_yang_table_1() -> None:
    '''
    Four coils reproduce Table 1 of Yang et al. (2025): plates 20.2, 14.6, 10.9 % and top 54.3 %.
    '''
    f = surface_flows(q_total=1.0, n_coils=4, zeta=ZETA)
    plates = [f[i]['top'] for i in range(3)]
    assert plates == pytest.approx([0.202, 0.146, 0.109], abs=1e-3)
    assert f[3]['top'] == pytest.approx(0.543, abs=1e-3)
    assert [c['outer'] for c in f] == pytest.approx([1.0, 0.798, 0.652, 0.543], abs=1e-3)


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6])
def test_mass_balance(n: int) -> None:
    '''
    Plate flows plus the top flow add up to the fan flow, and lower plates take more.
    '''
    f = surface_flows(q_total=1.0, n_coils=n, zeta=ZETA)
    plates = np.array([f[i]['top'] for i in range(n - 1)])
    assert plates.sum() + f[-1]['top'] == pytest.approx(1.0)
    assert np.all(np.diff(plates) < 0.0)


def test_bottom_modes() -> None:
    '''
    The bottom face of coil 1 has no flow in the Yang mode and the full flow with a base convector.
    '''
    assert surface_flows(q_total=2.0, n_coils=4, zeta=ZETA)[0]['bottom'] == 0.0
    assert surface_flows(q_total=2.0, n_coils=4, zeta=ZETA, bottom_mode='base_convector')[0]['bottom'] == 2.0
    with pytest.raises(ValueError):
        surface_flows(q_total=1.0, n_coils=4, zeta=ZETA, bottom_mode='other')


def test_scaled_geometry() -> None:
    '''
    Bore, annulus and radial channels get their own area and hydraulic diameter.
    '''
    g = scaled_channel_geometry(r_in=0.254, r_out=0.875, width=1.1, cover_radius=1.2255, plate_gap=0.08, top_gap=0.3, base_gap=0.1, is_top_coil=True)
    assert g['inner']['D'] == pytest.approx(0.508)
    assert g['outer']['D'] == pytest.approx(2.0 * (1.2255 - 0.875))
    assert g['outer']['area'] == pytest.approx(np.pi * (1.2255**2 - 0.875**2))
    assert g['bottom']['D'] == pytest.approx(0.08)
    assert g['top']['D'] == pytest.approx(0.6)
    assert g['base']['area'] == pytest.approx(2.0 * np.pi * 0.5645 * 0.1)
    with pytest.raises(ValueError):
        scaled_channel_geometry(r_in=0.254, r_out=1.3, width=1.1, cover_radius=1.2255, plate_gap=0.08, top_gap=0.3, base_gap=0.1, is_top_coil=False)


def test_laminar_floor() -> None:
    '''
    The floor only acts when Eq. 7 falls below the laminar value.
    '''
    assert laminar_floor(h=1.0, D=0.08, k_g=0.4, nusselt=7.54) == pytest.approx(7.54 * 0.4 / 0.08)
    assert laminar_floor(h=500.0, D=0.08, k_g=0.4, nusselt=7.54) == 500.0


def test_groups() -> None:
    '''
    The bottom face of coil 1 is the base group and the top face of the top coil is the top group.
    '''
    assert surface_group(coil=0, surface='bottom', n_coils=4) == 'base'
    assert surface_group(coil=1, surface='bottom', n_coils=4) == 'plates'
    assert surface_group(coil=3, surface='top', n_coils=4) == 'top'
    assert surface_group(coil=0, surface='inner', n_coils=4) == 'bore'


def test_flow_scale_and_psi_groups() -> None:
    '''
    flow_scale scales every velocity and a psi group multiplier only changes its own group.
    '''
    meshes = stack()
    base = make_surface_model(meshes=meshes, furnace=scaled())(coil=1, T_g=900.0)
    half = make_surface_model(meshes=meshes, furnace=scaled(flow_scale=0.5))(coil=1, T_g=900.0)
    boosted = make_surface_model(meshes=meshes, furnace=scaled(psi_groups={'bore': 2.0}))(coil=1, T_g=900.0)
    for s in ('inner', 'outer', 'bottom', 'top'):
        assert half[s]['u'] == pytest.approx(0.5 * base[s]['u'])
    assert boosted['inner']['h'] == pytest.approx(2.0 * base['inner']['h'])
    assert boosted['outer']['h'] == pytest.approx(base['outer']['h'])
    with pytest.raises(ValueError):
        make_surface_model(meshes=meshes, furnace=scaled(psi_groups={'walls': 2.0}))


def test_flow_report() -> None:
    '''
    The report has one row per surface; the base convector only carries flow in its mode.
    '''
    meshes = stack()
    rows = flow_report(meshes=meshes, furnace=scaled(), T_g=900.0)
    assert len(rows) == 16
    bottom = [r for r in rows if r['coil'] == 1 and r['surface'] == 'bottom'][0]
    assert bottom['h'] == 0.0
    rows = flow_report(meshes=meshes, furnace=scaled(bottom_mode='base_convector', flow_scale=0.3), T_g=900.0)
    bottom = [r for r in rows if r['coil'] == 1 and r['surface'] == 'bottom'][0]
    assert bottom['h'] > 0.0
    assert not any(r['flag'] for r in rows)

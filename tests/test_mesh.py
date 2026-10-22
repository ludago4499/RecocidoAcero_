'''
Tests for build_mesh using the four coils of Table 3 in Yang et al. (2025).

Table 3 does not give the inner radius, so we get it from the strip. Then we check
that the mesh weighs the same as the coil in the table.

Run from the project root with: python -m tests.test_mesh
'''

import numpy as np
import pytest

from bell_furnace.mesh import build_mesh

DENSITY = 7850.0  # steel density in kg/m3
N_R = 41          # radial nodes
N_Z = 31          # axial nodes

# Table 3:  outer diameter (mm), width (mm), thickness (mm), length (m), weight (kg)
TABLE_3 = {
    '1#': (1869.0, 1272.0, 1.50, 1634.0, 24490.0),
    '2#': (1867.0, 1250.0, 1.50, 1629.0, 23993.0),
    '3#': (1837.0, 1272.0, 1.48, 1593.0, 23557.0),
    '4#': (1761.0, 1272.0, 1.50, 1428.0, 21417.0),
}


def coil_dimensions(name: str) -> tuple[float, float, float, float]:
    '''
    Utility functions that get's the dimensions of one Table 3 coil in SI units.

    Args:
        name: coil name, for example '1#'

    Returns:
        Tuple (r_in, r_out, width, weight) with radii and width in m and weight in kg.
    '''
    diameter, width, thickness, length, weight = TABLE_3[name]
    r_out = diameter / 2000.0
    strip_area = length * thickness / 1000.0
    r_in = np.sqrt(r_out**2 - strip_area / np.pi) # Considering Area
    return float(r_in), r_out, width / 1000.0, weight


def make_mesh(name: str, geometry: str = 'axisymmetric', n_r: int = N_R, n_z: int = N_Z) -> dict:
    '''
    Build the mesh of one Table 3 coil.

    Args:
        name:     coil name, e.g. '1#'
        geometry: 'axisymmetric' or 'planar'
        n_r:      number of radial nodes
        n_z:      number of axial nodes

    Returns:
        The dict returned by build_mesh.
    '''
    r_in, r_out, width, _ = coil_dimensions(name)
    return build_mesh(r_in=r_in, r_out=r_out, width=width, n_r=n_r, n_z=n_z, geometry=geometry)


def test_inner_radius() -> None:
    '''
    The inner radius should be about 0.305 m

    '''
    for name in TABLE_3:
        r_in, _, _, _ = coil_dimensions(name)
        assert 0.28 < r_in < 0.33, name


def test_mass_matches_table() -> None:
    '''
    Mesh volume times density should give the weight in Table 3, within 1%.
    '''
    for name in TABLE_3:
        _, _, _, weight = coil_dimensions(name)
        mesh = make_mesh(name)
        mass = DENSITY * 2.0 * np.pi * mesh['volume'].sum()
        error = abs(mass - weight) / weight
        assert error < 0.01, f'{name}: mass {mass:.0f} kg vs {weight:.0f} kg'


def test_shapes() -> None:
    ''' 
    Every array should have the shape written in the docstring.
    '''
    for geometry in ['axisymmetric', 'planar']:
        mesh = make_mesh('1#', geometry)
        assert mesh['r'].shape == (N_R,)
        assert mesh['z'].shape == (N_Z,)
        assert mesh['volume'].shape == (N_Z, N_R)
        assert mesh['area_r'].shape == (N_Z, N_R - 1)
        assert mesh['area_z'].shape == (N_Z - 1, N_R)
        assert mesh['area_inner'].shape == (N_Z, N_R)
        assert mesh['area_outer'].shape == (N_Z, N_R)
        assert mesh['area_bottom'].shape == (N_Z, N_R)
        assert mesh['area_top'].shape == (N_Z, N_R)


def test_nodes_cover_the_coil() -> None:
    '''
    The first and last nodes should sit on the coil surfaces.
    '''
    for name in TABLE_3:
        r_in, r_out, width, _ = coil_dimensions(name)
        mesh = make_mesh(name)
        assert np.isclose(mesh['r'][0], r_in)
        assert np.isclose(mesh['r'][-1], r_out)
        assert np.isclose(mesh['z'][0], 0.0)
        assert np.isclose(mesh['z'][-1], width)


def test_totals_axisymmetric() -> None:
    '''
    Adding up all the small volumes and areas should give the exact values  
    of the whole coil considering Axisymmetric geometry.
    '''
    for name in TABLE_3:
        r_in, r_out, width, _ = coil_dimensions(name)
        mesh = make_mesh(name, 'axisymmetric')
        ring = 0.5 * (r_out**2 - r_in**2)
        assert np.isclose(mesh['volume'].sum(), ring * width)
        assert np.isclose(mesh['area_inner'].sum(), r_in * width)
        assert np.isclose(mesh['area_outer'].sum(), r_out * width)
        assert np.isclose(mesh['area_bottom'].sum(), ring)
        assert np.isclose(mesh['area_top'].sum(), ring)


def test_totals_planar() -> None:
    '''
    Same as the axisymmetric test, but using now planar geometry.
    '''
    for name in TABLE_3:
        r_in, r_out, width, _ = coil_dimensions(name)
        mesh = make_mesh(name, 'planar')
        thickness = r_out - r_in
        assert np.isclose(mesh['volume'].sum(), thickness * width)
        assert np.isclose(mesh['area_inner'].sum(), width)
        assert np.isclose(mesh['area_outer'].sum(), width)
        assert np.isclose(mesh['area_bottom'].sum(), thickness)
        assert np.isclose(mesh['area_top'].sum(), thickness)


def test_exposed_areas_only_on_edges() -> None:
    '''
    Gas only touches the surface, so exposed areas should be zero inside the coil.
    '''
    mesh = make_mesh('1#')
    assert np.all(mesh['area_inner'][:, 1:] == 0.0)   # only first column
    assert np.all(mesh['area_outer'][:, :-1] == 0.0)  # only last column
    assert np.all(mesh['area_bottom'][1:, :] == 0.0)  # only first row
    assert np.all(mesh['area_top'][:-1, :] == 0.0)    # only last row


def test_half_cells() -> None:
    '''
    Edge nodes own half a cell and corner nodes a quarter (planar,
    so a full cell is dr * dz).
    '''
    mesh = make_mesh('1#', 'planar')
    full = mesh['dr'] * mesh['dz']
    middle = mesh['volume'][N_Z // 2, N_R // 2]
    edge = mesh['volume'][0, N_R // 2]
    corner = mesh['volume'][0, 0]
    assert np.isclose(middle, full)
    assert np.isclose(edge, full / 2)
    assert np.isclose(corner, full / 4)


def test_bad_inputs() -> None:
    '''
    Collection of wrong inputs that should raise ValueError.
    '''
    with pytest.raises(ValueError):
        build_mesh(r_in=0.0, r_out=0.935, width=1.272, n_r=11, n_z=11)    # r_in not positive
    with pytest.raises(ValueError):
        build_mesh(r_in=0.305, r_out=0.2, width=1.272, n_r=11, n_z=11)    # r_out smaller than r_in
    with pytest.raises(ValueError):
        build_mesh(r_in=0.305, r_out=0.935, width=0.0, n_r=11, n_z=11)    # width not positive
    with pytest.raises(ValueError):
        build_mesh(r_in=0.305, r_out=0.935, width=1.272, n_r=2, n_z=11)   # too few nodes
    with pytest.raises(ValueError):
        build_mesh(r_in=0.305, r_out=0.935, width=1.272, n_r=11, n_z=11, geometry='square')


if __name__ == '__main__':
    print('Coil   r_in (m)   mesh mass (kg)   Table 3 (kg)   error')
    for name in TABLE_3:
        r_in, _, _, weight = coil_dimensions(name)
        mass = DENSITY * 2.0 * np.pi * make_mesh(name)['volume'].sum()
        print(f'{name:<6} {r_in:<10.4f} {mass:<16.0f} {weight:<14.0f} {100 * (mass - weight) / weight:.2f}%')
    pytest.main(['-v', __file__])

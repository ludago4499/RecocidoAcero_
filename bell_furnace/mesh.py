'''
Mesh of the steel coil r-z section and the explicit time step limit.

The coils can be represented in a 2D grid due to symmetry. the first
and last nodes lie on the coil surfaces, interior nodes own full control volumes, edge nodes own
half volumes and corner nodes own quarter volumes. 

Arrays have shape (n_z, n_r): row 0 is the bottom surface (z = 0) and column 0 is the inner surface (r = r_in).

All quantities are in SI units.
'''

import numpy as np


def coil_inner_diameter(*, mass: float, outer_diameter: float, width: float, density: float = 7850.0) -> float:
    '''
    Compute the inner diameter of a coil from its mass and outer dimensions.

    Uses mass = density * pi / 4 * (outer_diameter^2 - inner_diameter^2) * width.

    Args:
        mass:           coil mass in kg
        outer_diameter: coil outer diameter in m
        width:          coil width (strip width) in m
        density:        steel density in kg/m3

    Returns:
        Inner diameter in m.

    Raises:
        ValueError if any input is not positive or the mass is too large for the outer dimensions.
    '''
    if min(mass, outer_diameter, width, density) <= 0.0:
        raise ValueError('mass, outer_diameter, width and density must be positive')

    d_squared = outer_diameter**2 - 4.0 * mass / (np.pi * density * width)
    if d_squared <= 0.0:
        raise ValueError('mass is too large for the given outer diameter and width')
    return float(np.sqrt(d_squared))


def build_mesh(*, r_in: float, r_out: float, width: float, n_r: int, n_z: int, geometry: str = 'axisymmetric') -> dict[str, np.ndarray | float | int | str]:
    '''
    Build the vertex-centred mesh of one coil section.

    Args:
        r_in:     coil inner radius in m
        r_out:    coil outer radius in m
        width:    coil width (axial length) in m
        n_r:      number of nodes in the radial direction
        n_z:      number of nodes in the axial direction
        geometry: 'axisymmetric' (per radian) or 'planar' (per unit depth, as in the paper)

    Returns:
        Dict with keys:
            r, z:                     1D node coordinates in m, shapes (n_r,) and (n_z,)
            R, Z:                     2D node coordinates in m, shape (n_z, n_r)
            dr, dz:                   node spacings in m
            volume:                   control volume of each node, shape (n_z, n_r)
            area_r:                   radial face areas between neighbours, shape (n_z, n_r - 1)
            area_z:                   axial face areas between neighbours, shape (n_z - 1, n_r)
            area_inner, area_outer:   exposed area on the inner and outer surfaces, shape (n_z, n_r)
            area_bottom, area_top:    exposed area on the bottom and top surfaces, shape (n_z, n_r)
            r_in, r_out, width:       the input dimensions in m
            n_r, n_z, geometry:       the input resolution and geometry
        Volumes are in m3 and areas in m2 per radian for 'axisymmetric', and m2 and m per unit
        depth for 'planar'. Exposed areas are zero away from their surface.

    Raises:
        ValueError if the dimensions are not positive, r_in >= r_out, n_r or n_z is below 3,
        or geometry is not planar or axisymmetric.
    '''
    if r_in <= 0.0 or width <= 0.0:
        raise ValueError('r_in and width must be positive in m')
    if r_out <= r_in:
        raise ValueError('r_out must be larger than r_in')
    if n_r < 3 or n_z < 3:
        raise ValueError('n_r and n_z must be at least 3')
    if geometry not in ('axisymmetric', 'planar'):
        raise ValueError("geometry must be 'axisymmetric' or 'planar'")

    r = np.linspace(r_in, r_out, n_r)
    z = np.linspace(0.0, width, n_z)
    dr = (r_out - r_in) / (n_r - 1)
    dz = width / (n_z - 1)

    #  neighbors
    r_w = np.maximum(r - 0.5 * dr, r_in)
    r_e = np.minimum(r + 0.5 * dr, r_out)
    dz_cell = np.full(n_z, dz)

    # Boundary scenario
    dz_cell[[0, -1]] = 0.5 * dz 
    r_face = 0.5 * (r[:-1] + r[1:])

    if geometry == 'axisymmetric':
        ring = 0.5 * (r_e**2 - r_w**2)  # annulus area per radian, m2
        radial_face = r_face  # face height multiplier per radian, m
        inner_factor, outer_factor = r_in, r_out # for surface area
    else:
        ring = r_e - r_w  # m per unit depth
        radial_face = np.ones(n_r - 1)
        inner_factor, outer_factor = 1.0, 1.0 #

    volume = np.outer(dz_cell, ring)
    area_r = np.outer(dz_cell, radial_face)
    area_z = np.tile(ring, (n_z - 1, 1)) # stack n_z rows

    area_inner = np.zeros((n_z, n_r))
    area_outer = np.zeros((n_z, n_r))
    area_bottom = np.zeros((n_z, n_r))
    area_top = np.zeros((n_z, n_r))
    area_inner[:, 0] = inner_factor * dz_cell
    area_outer[:, -1] = outer_factor * dz_cell
    area_bottom[0, :] = ring
    area_top[-1, :] = ring

    R, Z = np.meshgrid(r, z)

    return {
        'r': r, 
        'z': z,
        'R': R,
        'Z': Z,
        'dr': dr,
        'dz': dz,
        'volume': volume,
        'area_r': area_r,
        'area_z': area_z,
        'area_inner': area_inner,
        'area_outer': area_outer,
        'area_bottom': area_bottom,
        'area_top': area_top,
        'r_in': r_in,
        'r_out': r_out,
        'width': width,
        'n_r': n_r,
        'n_z': n_z,
        'geometry': geometry,
    }


def build_stack(*, coils: list[dict[str, float]], n_r: int, n_z: int, geometry: str = 'axisymmetric') -> list[dict[str, np.ndarray | float | int | str]]:
    '''
    Build one mesh per coil in a stack, ordered bottom to top.

    Args:
        coils:    list of dicts with keys inner_diameter, outer_diameter and width in m, bottom coil first
        n_r:      numbe r of radial nodes per coil
        n_z:      number of axial nodes per coil
        geometry: 'axisymmetric' or 'planar'

    Returns:
        List of mesh dicts as returned by build_mesh, one per coil, bottom coil first.

    Raises:
        ValueError if coils is empty or a coil dict is missing a key; errors from build_mesh pass through.
    '''
    if not coils:
        raise ValueError('coils must contain at least one coil')

    meshes = []
    for i, coil in enumerate(coils):
        missing = {'inner_diameter', 'outer_diameter', 'width'} - coil.keys()
        if missing:
            raise ValueError(f'coil {i} is missing keys: {sorted(missing)}')
        meshes.append(build_mesh(r_in=0.5 * coil['inner_diameter'], r_out=0.5 * coil['outer_diameter'], width=coil['width'], n_r=n_r, n_z=n_z, geometry=geometry))
    return meshes


def stable_time_step(*, mesh: dict[str, np.ndarray | float | int | str], rho_c: np.ndarray, G_r: np.ndarray, G_z: np.ndarray, h: dict[str, float], safety: float = 0.9) -> float:
    '''
    Compute the largest stable time step of the explicit scheme.

    For each node the explicit update is T_new = T + dt / C * (sum of G * (T_neighbour - T) +
    sum of h * A * (T_g - T)) with C = rho_c * volume.

    Args:
        mesh:   mesh dict from build_mesh
        rho_c:  volumetric heat capacity of each node in J/(m3 K), shape (n_z, n_r)
        G_r:    radial face conductances in W/K, shape (n_z, n_r - 1)
        G_z:    axial face conductances in W/K, shape (n_z - 1, n_r)
        h:      surface heat transfer coefficients in W/(m2 K) with keys inner, outer, bottom, top
        safety: factor in (0, 1] applied to the limit

    Returns:
        Stable time step in s.

    Raises:
        ValueError if an array has the wrong shape, h is missing a surface or has a negative value,
        rho_c is not positive, or safety is outside (0, 1].
    '''
    n_z, n_r = mesh['n_z'], mesh['n_r']
    if rho_c.shape != (n_z, n_r):
        raise ValueError(f'rho_c must have shape {(n_z, n_r)}')
    if G_r.shape != (n_z, n_r - 1) or G_z.shape != (n_z - 1, n_r):
        raise ValueError('G_r or G_z has the wrong shape for this mesh')
    if np.any(rho_c <= 0.0):
        raise ValueError('rho_c must be positive')
    missing = {'inner', 'outer', 'bottom', 'top'} - h.keys()
    if missing:
        raise ValueError(f'h is missing surfaces: {sorted(missing)}')
    if any(value < 0.0 for value in h.values()):
        raise ValueError('h must not be negative')
    if not 0.0 < safety <= 1.0:
        raise ValueError('safety must be in (0, 1]')

    g_sum = np.zeros((n_z, n_r))
    g_sum[:, :-1] += G_r
    g_sum[:, 1:] += G_r
    g_sum[:-1, :] += G_z
    g_sum[1:, :] += G_z
    g_sum += h['inner'] * mesh['area_inner'] + h['outer'] * mesh['area_outer'] + h['bottom'] * mesh['area_bottom'] + h['top'] * mesh['area_top']

    capacity = rho_c * mesh['volume']
    with np.errstate(divide='ignore'):
        dt_node = np.where(g_sum > 0.0, capacity / g_sum, np.inf)
    return safety * float(np.min(dt_node))
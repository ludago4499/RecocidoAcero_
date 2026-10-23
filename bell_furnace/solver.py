'''
Explicit finite-volume solver for the temperature field of the coils in a bell-type annealing furnace.

Constants that solver.py needs

    C    = rho * cp * volume                    heat capacity of the node in J/K
    G    = lambda_face * face_area / spacing    conductance between two neighbouring nodes in W/K
    h A  = surface coefficient times exposed area in W/K (zero if the node is not on that surface)

    Forward Euler in time 
        T_new = T + dt / C * (conduction + convection)
    
    For stability we take max time = 30, if not we compute conduction + convection and take the minimum that 
    won't break stability  for any node ( as in the paper)

Units
    SI everywhere, temperatures in K, time in s. In the axisymmetric mesh, volumes and areas are per
    radian.
'''

from types import FunctionType  # type of any function made with def or lambda, used only in type hints

import numpy as np

try:
    # works when bell_furnace is imported as a package (import bell_furnace.solver)
    from .convection import channel_geometry, composite_coefficient, convection_coefficient, simple_channel_geometry
    from .flow import actual_flow_rate, surface_flows
    from .mesh import stable_time_step
    from .properties import hydrogen_properties, steel_properties
    from .radial_conductivity import effective_radial_conductivity
except ImportError:
    # works when running a script from inside the bell_furnace folder (python solver.py)
    from convection import channel_geometry, composite_coefficient, convection_coefficient, simple_channel_geometry
    from flow import actual_flow_rate, surface_flows
    from mesh import stable_time_step
    from properties import hydrogen_properties, steel_properties
    from radial_conductivity import effective_radial_conductivity


SURFACES = ('inner', 'outer', 'bottom', 'top')

# Temporary constant values used by the placeholder functions, only to get first results
PLACEHOLDER_H = 100.0  # W/(m2 K)
PLACEHOLDER_LAMBDA_R = 3.0  # W/(m K)



def gas_temperature(*, time: float, schedule_time: np.ndarray, schedule_temperature: np.ndarray) -> float:
    '''
    Interpolate the protective gas temperature at a given time.

    The schedule is a list of (time, temperature) points. 
    Args:
        time:                 time in s
        schedule_time:        times of the schedule points in s, increasing
        schedule_temperature: gas temperatures at those points in K

    Returns:
        Gas temperature in K.
    '''
    return float(np.interp(time, schedule_time, schedule_temperature))


def trapezoid_schedule(*, T_start: float, T_hold: float, T_end: float, heating_time: float, holding_time: float, cooling_time: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Build a simple heat-hold-cool gas temperature schedule.

    Four points: linear heating from T_start to T_hold, constant hold, linear cooling to T_end. It represents
    a trapezoid.

    Args:
        T_start:      gas temperature at time 0 in K
        T_hold:       holding temperature in K
        T_end:        gas temperature at the end of cooling in K
        heating_time: duration of the heating ramp in s
        holding_time: duration of the hold in s
        cooling_time: duration of the cooling ramp in s

    Returns:
        Tuple (schedule_time, schedule_temperature) of arrays in s and K.

    Raises:
        ValueError if a duration is not positive or a temperature is not positive.
    '''
    if min(heating_time, holding_time, cooling_time) <= 0.0:
        raise ValueError('heating_time, holding_time and cooling_time must be positive in s')
    if min(T_start, T_hold, T_end) <= 0.0:
        raise ValueError('temperatures must be positive in K')

    t1 = heating_time
    t2 = t1 + holding_time
    t3 = t2 + cooling_time
    schedule_time = np.array([0.0, t1, t2, t3])
    schedule_temperature = np.array([T_start, T_hold, T_hold, T_end])
    return schedule_time, schedule_temperature


# Placeholder values for tests

def placeholder_h(*, coil: int, T_g: float, time: float) -> dict[str, float]:
    '''
    Return constant convection coefficients on the four coil surfaces.

    The bottom face of the bottom coil (coil 0) sits on the solid base convector and has no gas flow
    This is a placeholder that is still missing to be merged with team's work.

    Args:
        coil: index of the coil in the stack, 0 is the bottom coil
        T_g:  gas temperature in K (unused here, the real correlation needs it)
        time: time in s (unused here)

    Returns:
        Dict with keys inner, outer, bottom, top and values in W/(m2 K).
    '''
    h = {}
    for surface in SURFACES:
        h[surface] = PLACEHOLDER_H
    if coil == 0:
        h['bottom'] = 0.0
    return h


def placeholder_lambda_r(*, coil: int, T: np.ndarray, T_g: float) -> np.ndarray:
    '''
    Return a constant radial conductivity for every node. Placeholder a nd needs to be 
    merged with team's work.

    Args:
        coil: index of the coil in the stack (unused here)
        T:    node temperatures in K, shape (n_z, n_r)
        T_g:  gas temperature in K (unused here, Eq. 17 switches form at 800 K)

    Returns:
        Radial conductivity in W/(m K), same shape as T.
    '''
    return np.full_like(T, PLACEHOLDER_LAMBDA_R, dtype=float)


# ------------------------------------------------------------------------------------------------
# Model physics
# ------------------------------------------------------------------------------------------------

def make_lambda_r_function(*, strip_thickness: list[float], contact: dict[str, float]) -> FunctionType:
    '''
    Build the radial conductivity plug-in from Eqs. 9-17. Wrap around function.

    Args:
        strip_thickness: strip thickness S of each coil in m
        contact:         contact parameters with keys pressure, hardness, roughness, tan_theta,
                         epsilon, oxide_thickness, k_oxide, eta (see furnace_parameters.CONTACT)

    Returns:
        Function (*, coil, T, T_g) returning the radial conductivity in W/(m K) with the shape of T.

    Raises:
        ValueError if strip_thickness is empty or has a non-positive value.
    '''
    thickness = []
    for s in strip_thickness:
        thickness.append(float(s))
    if not thickness or min(thickness) <= 0.0:
        raise ValueError('strip_thickness must contain positive values in m')
    contact = dict(contact)

    def lambda_r_function(*, coil: int, T: np.ndarray, T_g: float) -> np.ndarray:
        _, _, k_s = steel_properties(T)
        _, _, k_g, _ = hydrogen_properties(T)
        return effective_radial_conductivity(T_m=T, T_g=T_g, k_s=k_s, k_g=k_g, s=thickness[coil], **contact)

    return lambda_r_function

def make_h_function(*, meshes: list[dict], furnace: dict) -> FunctionType:
    '''
    Build the surface coefficient plug-in from Eqs. 5-8 (flow.py and convection.py).

    Args:
        meshes:  mesh dicts from mesh.build_stack, bottom coil first
        furnace: furnace parameters (see furnace_parameters.FURNACE)

    Returns:
        Function (*, coil, T_g, time) returning h in W/(m2 K) on inner, outer, bottom and top.

    Raises:
        ValueError if furnace['channel_model'] is not 'simple' or 'detailed'.
    '''
    n_coils = len(meshes)

    # Share of the fan flow that reaches each surface of each coil (Eqs. 5-6)
    fractions = surface_flows(q_total=1.0, n_coils=n_coils, zeta=furnace['zeta'])

    # Channel area, D and L of each surface of each coil
    channels = []
    for i, mesh in enumerate(meshes):
        if furnace['channel_model'] == 'simple':
            channel = simple_channel_geometry(area=furnace['channel_area'], D=furnace['channel_D'], L=furnace['channel_L'])
        elif furnace['channel_model'] == 'detailed':
            channel = channel_geometry(r_in=mesh['r_in'], r_out=mesh['r_out'], width=mesh['width'], cover_radius=0.5 * furnace['cover_diameter'], plate_gap=furnace['plate_gap'], top_gap=furnace['top_gap'], is_top_coil=(i == n_coils - 1))
        else:
            raise ValueError("channel_model must be 'simple' or 'detailed'")
        channels.append(channel)

    def h_function(*, coil: int, T_g: float, time: float) -> dict[str, float]:
        q_actual = actual_flow_rate(q_normal=furnace['q_normal'], T_g=T_g, pressure=furnace['pressure'])
        rho_g, mu_g, k_g, _ = hydrogen_properties(T_g)
        h = {}
        for surface in SURFACES:
            q_surface = q_actual * fractions[coil][surface]
            if q_surface == 0.0:
                h[surface] = 0.0  # no gas flow on this surface
                continue
            channel = channels[coil][surface]
            u_g = q_surface / channel['area']
            h_conv = convection_coefficient(u_g=u_g, D=channel['D'], L=channel['L'], k_g=float(k_g), mu_g=float(mu_g), rho_g=float(rho_g), psi=furnace['psi'])
            h[surface] = composite_coefficient(h=h_conv, T_g=T_g, xi=furnace['xi'])
        return h

    return h_function

# ------------------------------------------------------------------------------------------------
# Building blocks of one time step
# ------------------------------------------------------------------------------------------------

def face_conductances(*, mesh: dict, lambda_r: np.ndarray, lambda_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the thermal conductance between every pair of neighbouring nodes. We base ourselves on the figure 5.

    Args:
        mesh:     mesh dict from mesh.build_mesh
        lambda_r: radial conductivity of each node in W/(m K), shape (n_z, n_r)
        lambda_z: axial conductivity of each node in W/(m K), shape (n_z, n_r)

    Returns:
        Tuple (G_r, G_z): radial conductances in W/K with shape (n_z, n_r - 1), where G_r[i, j]
        connects nodes (i, j) and (i, j + 1), and axial conductances in W/K with shape
        (n_z - 1, n_r), where G_z[i, j] connects nodes (i, j) and (i + 1, j).
    '''
    a, b = lambda_r[:, :-1], lambda_r[:, 1:]
    lambda_r_face = 2.0 * a * b / (a + b)
    G_r = lambda_r_face * mesh['area_r'] / mesh['dr']

    a, b = lambda_z[:-1, :], lambda_z[1:, :]
    lambda_z_face = 2.0 * a * b / (a + b)
    G_z = lambda_z_face * mesh['area_z'] / mesh['dz']

    return G_r, G_z


def node_heat_rates(*, T: np.ndarray, T_g: float, mesh: dict, G_r: np.ndarray, G_z: np.ndarray, h: dict[str, float]) -> np.ndarray:
    '''
    Compute the net heat entering every node by conduction and convection.

    Conduction: Moves energy c onsidering the conductance G
    Convection: Adds energy due to the gas on the faces and with the convection coeff h.

    Args:
        T:    node temperatures in K, shape (n_z, n_r)
        T_g:  gas temperature in K
        mesh: mesh dict from mesh.build_mesh
        G_r:  radial conductances in W/K, shape (n_z, n_r - 1)
        G_z:  axial conductances in W/K, shape (n_z - 1, n_r)
        h:    convection coefficients in W/(m2 K) with keys inner, outer, bottom, top

    Returns:
        Net heat rate into each node in W (per radian for the axisymmetric mesh), shape (n_z, n_r).
        Positive means the node heats up.
    '''
    q = np.zeros_like(T)

    # Radial conduction: heat flowing from column j + 1 into column j
    flow_r = G_r * (T[:, 1:] - T[:, :-1])
    q[:, :-1] += flow_r
    q[:, 1:] -= flow_r

    # Axial conduction: heat flowing from row i + 1 into row i
    flow_z = G_z * (T[1:, :] - T[:-1, :])
    q[:-1, :] += flow_z
    q[1:, :] -= flow_z

    # Convection from the gas on each surface
    for surface in SURFACES:
        q += h[surface] * mesh[f'area_{surface}'] * (T_g - T)

    return q


def explicit_step(*, T: np.ndarray, T_g: float, mesh: dict, coil: int, h: dict[str, float], lambda_r_function: FunctionType, dt_max: float, safety: float, T_g_peak: float | None = None) -> tuple[np.ndarray, float]:
    '''
    Advance the coil temperature by one explicit time step. 
    At each step:
        1. Calculate steel properties at current temperature
        2. Radial conductivity lambda r
        3. face conductanes

    Order of operations:
        1. steel properties rho, cp and k at the current temperatures (steel.csv)
        2. radial conductivity from lambda_r_function, axial conductivity = k of steel
        3. face conductances G_r and G_z
        4. largest stable step from mesh.stable_time_step; the step is min(dt_max, stable step)
        5. heat rates Q and update T_new = T + dt Q / (rho cp volume)

    Args:
        T:                 node temperatures in K, shape (n_z, n_r)
        T_g:               gas temperature in K, constant during the step
        mesh:              mesh dict from mesh.build_mesh
        coil:              index of the coil in the stack (passed to lambda_r_function)
        h:                 convection coefficients in W/(m2 K) with keys inner, outer, bottom, top
        lambda_r_function: function (*, coil, T, T_g) returning the radial conductivity in W/(m K)
        dt_max:            largest step allowed by the caller in s
        safety:            factor in (0, 1] applied to the stability limit
        T_g_peak:          highest gas temperature reached so far in K, passed to lambda_r_function
                           as its T_g; None uses T_g. Eq. 17 uses the emulsion form only while
                           heating below 800 K: once the gas has passed 800 K the emulsion is gone
                           and the oxide form holds for the rest of the cycle, cooling included.

    Returns:
        Tuple (T_new, dt): new temperatures in K and the step actually taken in s.

    Raises:
        ValueError if lambda_r_function returns the wrong shape or a non-positive value; errors from
        steel_properties (temperature outside the table) and stable_time_step pass through.
    '''
    rho, cp, k = steel_properties(T)
    rho_c = rho * cp

    T_g_regime = T_g if T_g_peak is None else max(T_g, T_g_peak)
    lambda_r = np.asarray(lambda_r_function(coil=coil, T=T, T_g=T_g_regime), dtype=float)

    if lambda_r.shape != T.shape:
        raise ValueError(f'lambda_r_function must return shape {T.shape}')
    if np.any(lambda_r <= 0.0):
        raise ValueError('radial conductivity must be positive')

    G_r, G_z = face_conductances(mesh=mesh, lambda_r=lambda_r, lambda_z=k)

    dt_stable = stable_time_step(mesh=mesh, rho_c=rho_c, G_r=G_r, G_z=G_z, h=h, safety=safety)
    dt = min(dt_max, dt_stable) #  max time step is 30s.

    Q = node_heat_rates(T=T, T_g=T_g, mesh=mesh, G_r=G_r, G_z=G_z, h=h)
    T_new = T + dt * Q / (rho_c * mesh['volume'])
    return T_new, dt


# ------------------------------------------------------------------------------------------------
# Time loop
# ------------------------------------------------------------------------------------------------

def simulate_coil(*, mesh: dict, coil: int, T_initial: float, schedule_time: np.ndarray, schedule_temperature: np.ndarray, h_function: FunctionType = placeholder_h, lambda_r_function: FunctionType = placeholder_lambda_r, dt_max: float = 30.0, save_interval: float = 600.0, safety: float = 0.9) -> dict:
    '''
    Simulate one coil from time 0 to the end of the gas schedule.

    A step is shortened when the stability limit is smaller, and also so that it lands exactly on the next save time. 
    Results are stored only every save_interval to improve memory.

    Args:
        mesh:                 mesh dict of this coil from mesh.build_mesh
        coil:                 index of the coil in the stack, 0 is the bottom coil
        T_initial:            uniform initial coil temperature in K
        schedule_time:        times of the gas schedule in s, starting at 0 and increasing
        schedule_temperature: gas temperatures of the schedule in K
        h_function:           function (*, coil, T_g, time) returning h per surface in W/(m2 K)
        lambda_r_function:    function (*, coil, T, T_g) returning the radial conductivity in W/(m K)
        dt_max:               largest time step in s
        save_interval:        time between saved results in s
        safety:               factor in (0, 1] applied to the stability limit

    Returns:
        Dict with keys:
            time:     saved times in s, shape (n_save,)
            T_gas:    gas temperature at the saved times in K, shape (n_save,)
            T:        temperature field at the saved times in K, shape (n_save, n_z, n_r)
            T_min:    cold spot temperature in K, shape (n_save,)
            T_max:    hot spot temperature in K, shape (n_save,)
            T_mean:   volume-weighted mean coil temperature in K, shape (n_save,)
            n_steps:  number of time steps taken
            dt_min:   smallest step taken because of stability in s (shows if dt_max was limiting)

    Raises:
        ValueError if the schedule is invalid, or dt_max, save_interval or T_initial is not positive.
    '''
    schedule_time = np.asarray(schedule_time, dtype=float)
    schedule_temperature = np.asarray(schedule_temperature, dtype=float)

    if schedule_time.ndim != 1 or schedule_time.shape != schedule_temperature.shape or schedule_time.size < 2:
        raise ValueError('schedule_time and schedule_temperature must be 1D arrays of the same length, at least 2')
    if schedule_time[0] != 0.0 or np.any(np.diff(schedule_time) <= 0.0):
        raise ValueError('schedule_time must start at 0 and be strictly increasing')
    if dt_max <= 0.0 or save_interval <= 0.0:
        raise ValueError('dt_max and save_interval must be positive in s')
    if T_initial <= 0.0:
        raise ValueError('T_initial must be positive in K')

    t_end = schedule_time[-1]
    volume = mesh['volume']
    T = np.full((mesh['n_z'], mesh['n_r']), float(T_initial)) # fill function

    saved_time, saved_gas, saved_T = [], [], []

    def save(t: float) -> None:
        # appends the current state to the result lists
        saved_time.append(t)
        saved_gas.append(gas_temperature(time=t, schedule_time=schedule_time, schedule_temperature=schedule_temperature))
        saved_T.append(T.copy())

    t = 0.0
    next_save = min(save_interval, t_end)
    n_steps = 0
    T_g_peak = 0  # highest gas temperature so far, kelvin
    dt_min = np.inf
    save(t)

    while t < t_end: # doesn't use n_steps for now? 
        T_g = gas_temperature(time=t, schedule_time=schedule_time, schedule_temperature=schedule_temperature)
        h = h_function(coil=coil, T_g=T_g, time=t)

        limit = min(dt_max, next_save - t) # limit for dt
        T_g_peak = max(T_g_peak, T_g)
        T, dt = explicit_step(T=T, T_g=T_g, mesh=mesh, coil=coil, h=h, lambda_r_function=lambda_r_function, dt_max=limit, safety=safety, T_g_peak=T_g_peak)

        # after explicit_step
        t += dt 
        n_steps += 1
        if dt < limit:
            dt_min = min(dt_min, dt) # make sure to not pass time limit for errors in trapezoid

        # snap to next save
        if t >= next_save - 1e-9:
            t = next_save
            save(t)
            next_save = min(next_save + save_interval, t_end)

    T_hist = np.array(saved_T)
    return {
        'time': np.array(saved_time),
        'T_gas': np.array(saved_gas),
        'T': T_hist,
        'T_min': T_hist.min(axis=(1, 2)),
        'T_max': T_hist.max(axis=(1, 2)),
        'T_mean': (T_hist * volume).sum(axis=(1, 2)) / volume.sum(),
        'n_steps': n_steps,
        'dt_min': float(dt_min),
    }


def simulate_stack(*, meshes: list[dict], T_initial: float, schedule_time: np.ndarray, schedule_temperature: np.ndarray, h_function: FunctionType = placeholder_h, lambda_r_function: FunctionType = placeholder_lambda_r, dt_max: float = 30.0, save_interval: float = 600.0, safety: float = 0.9) -> list[dict]:
    '''
    Simulate every coil of the stack with the same gas temperature history (for now) .

    Args:
        meshes:               list of mesh dicts from mesh.build_stack, bottom coil first
        T_initial:            uniform initial temperature of all coils in K
        schedule_time:        times of the gas schedule in s
        schedule_temperature: gas temperatures of the schedule in K
        h_function:           function (*, coil, T_g, time) returning h per surface in W/(m2 K)
        lambda_r_function:    function (*, coil, T, T_g) returning the radial conductivity in W/(m K)
        dt_max:               largest time step in s
        save_interval:        time between saved results in s
        safety:               factor in (0, 1] applied to the stability limit

    Returns:
        List with one result dict per coil (see simulate_coil), bottom coil first.
    '''
    results = []
    for i, mesh in enumerate(meshes):
        result = simulate_coil(mesh=mesh, coil=i, T_initial=T_initial, schedule_time=schedule_time, schedule_temperature=schedule_temperature, h_function=h_function, lambda_r_function=lambda_r_function, dt_max=dt_max, save_interval=save_interval, safety=safety)
        results.append(result)
    return results


# ------------------------------------------------------------------------------------------------
# Output metrics
# ------------------------------------------------------------------------------------------------

def time_to_reach(*, time: np.ndarray, values: np.ndarray, target: float, rising: bool = True) -> float:
    '''
    Find the first time a temperature series crosses a target, with linear interpolation.

    It checks if the gas temp is rising.

    Args:
        time:   times in s, increasing
        values: temperatures in K at those times
        target: target temperature in K
        rising: true to look for values going up to the target, false for values going down

    Returns:
        Crossing time in s, or nan if the target is never reached.
    '''
    reached = values >= target if rising else values <= target
    if not np.any(reached):
        return float('nan')
    k = int(np.argmax(reached))  # first index where the target is reached
    if k == 0:
        return float(time[0])
    # Linear interpolation to get the correct time.
    fraction = (target - values[k - 1]) / (values[k] - values[k - 1])
    return float(time[k - 1] + fraction * (time[k] - time[k - 1]))

if __name__ == '__main__':
    try:
        # python -m bell_furnace.solver (from the project root)
        from .furnace_parameters import CONTACT, FURNACE, TABLE_3
        from .mesh import build_stack, coil_inner_diameter
    except ImportError:
        # python solver.py (from inside the bell_furnace folder)
        from furnace_parameters import CONTACT, FURNACE, TABLE_3
        from mesh import build_stack, coil_inner_diameter

    coils = []
    strip_thickness = []
    for c in TABLE_3:
        inner_diameter = coil_inner_diameter(mass=c['mass'], outer_diameter=c['outer_diameter'], width=c['width'])
        coils.append({'inner_diameter': inner_diameter, 'outer_diameter': c['outer_diameter'], 'width': c['width']})
        strip_thickness.append(c['thickness'])
    meshes = build_stack(coils=coils, n_r=21, n_z=31)
    h_function = make_h_function(meshes=meshes, furnace=FURNACE)
    lambda_r_function = make_lambda_r_function(strip_thickness=strip_thickness, contact=CONTACT)

    # Stand-in schedule: 12 h heating to 973 K, 10 h hold, 20 h cooling to 423 K
    hour = 3600.0
    schedule_time, schedule_temperature = trapezoid_schedule(T_start=303.15, T_hold=973.15, T_end=423.15, heating_time=12 * hour, holding_time=10 * hour, cooling_time=20 * hour)

    # Coefficients at two scenarios: cold and a hot gas temperature, to check their order of magnitude
    for T_g in (500.0, 973.15):
        print(f'T_g = {T_g} K')
        for i in range(len(meshes)):
            h = h_function(coil=i, T_g=T_g, time=0.0)
            lam = lambda_r_function(coil=i, T=np.array([[T_g]]), T_g=T_g)[0, 0]
            print(f"  {i + 1}#  h inner {h['inner']:6.1f}  outer {h['outer']:6.1f}  bottom {h['bottom']:6.1f}  top {h['top']:6.1f} W/(m2 K)   lambda_r {lam:5.2f} W/(m K)")

    results = simulate_stack(meshes=meshes, T_initial=303.15, schedule_time=schedule_time, schedule_temperature=schedule_temperature, h_function=h_function, lambda_r_function=lambda_r_function)

    print('coil  ID [m]  hot/cold at end of heating [K]  cold spot within 20 K of hold [h]  steps')
    for i, (coil, result) in enumerate(zip(coils, results)):
        k = int(np.searchsorted(result['time'], 12 * hour))
        t_soak = time_to_reach(time=result['time'], values=result['T_min'], target=900)
        print(f"{i + 1}#    {coil['inner_diameter']:.3f}   {result['T_max'][k]:.1f} / {result['T_min'][k]:.1f}                   {t_soak / hour:.2f}                             {result['n_steps']}")

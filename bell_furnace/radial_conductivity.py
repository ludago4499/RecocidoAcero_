
# radial_conductivity.py


import numpy as np

SIGMA = 5.670374419e-8


def effective_radial_conductivity(
    *,
    T_m,
    T_g,
    k_s,
    k_g,
    s,
    pressure,
    hardness,
    roughness,
    tan_theta,
    epsilon,
    oxide_thickness,
    k_oxide,
    eta
):
    """
    Calculate the effective radial conductivity of a steel coil.

    Based on equations (9)-(17) of Yang et al. (2025).

    Args:
        T_m: mean steel temperature (K).
        T_g: protective gas temperature (K).
        k_s: steel thermal conductivity (W/m K).
        k_g: hydrogen thermal conductivity (W/m K).
        s: steel strip thickness (m).
        pressure: contact pressure (MPa).
        hardness: contact material hardness (MPa).
        roughness: surface roughness (m).
        tan_theta: mean surface slope (-).
        epsilon: emissivity (-).
        oxide_thickness: oxide layer thickness (m).
        k_oxide: oxide conductivity (W/m K).
        eta: emulsion correction factor (-).

    Returns:
        Effective radial conductivity (W/m K).
        Supports scalars and NumPy arrays.
    """

    # Validate material and contact parameters
    values = np.array([
        s, pressure, hardness, roughness, tan_theta,
        epsilon, oxide_thickness, k_oxide, eta
    ], dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("Parameters must be finite")

    if (s <= 0 or pressure < 0 or hardness <= 0
            or roughness <= 0 or tan_theta <= 0
            or oxide_thickness < 0 or k_oxide <= 0
            or eta <= 0 or not 0 <= epsilon <= 1):
        raise ValueError("Invalid material or contact parameters")

    T_m, T_g, k_s, k_g = np.broadcast_arrays(
        np.asarray(T_m, dtype=float),
        np.asarray(T_g, dtype=float),
        np.asarray(k_s, dtype=float),
        np.asarray(k_g, dtype=float)
    )

    if (not all(np.all(np.isfinite(x)) for x in
                (T_m, T_g, k_s, k_g))
            or np.any(T_m <= 0) or np.any(T_g <= 0)
            or np.any(k_s <= 0) or np.any(k_g <= 0)):
        raise ValueError("Temperatures and conductivities must be positive and finite")

    # Eq. (14): contact area fraction
    phi = pressure / (hardness + pressure)

    # Eq. (15): gap between steel layers
    # Pressure must be expressed in MPa
    b = 42.7e-6 * np.exp(-0.05 * pressure)

    # Eq. (9): resistance of the steel strip
    R_steel = s / k_s

    # Eq. (10): oxide layer resistance
    R_oxide = oxide_thickness / k_oxide

    # Eq. (11): radiation conductance
    h_rad = (
        4 * (1 - phi) * epsilon * SIGMA * T_m**3
        / (2 - epsilon)
    )

    # Eq. (12): gas conductance
    h_gas = (1 - phi) * k_g / b

    # Eq. (13): contact conductance
    h_contact = (
        1.13 * k_s * tan_theta
        / roughness * phi**0.94
    )

    # Three parallel heat-transfer mechanisms
    R_interface = 1 / (h_rad + h_gas + h_contact)

    # Eq. (16): high-temperature total resistance
    R_high = R_steel + R_oxide + R_interface

    # Eq. (17): low-temperature denominator
    R_low = R_steel + R_interface

    # Eq. (17): effective radial conductivity
    k_high = (s + b + oxide_thickness) / R_high
    k_low = eta * (s + b) / R_low

    k_radial = np.where(T_g >= 800, k_high, k_low)

    if np.any(~np.isfinite(k_radial)) or np.any(k_radial <= 0):
        raise ValueError("Invalid radial conductivity result")

    return float(k_radial) if k_radial.ndim == 0 else k_radial

'''
Input parameters of the heat transfer model, separated by origin.

Two cases:
    paper:   CONTACT, FURNACE, TABLE_3 (Yang et al. 2025), generic steel
    ternium: TERNIUM (raw data given by Ternium, in SI), FURNACE_TERNIUM, CONTACT_TERNIUM,
             TERNIUM_COILS and SCHEDULE_TERNIUM; grades SAE 1006 and SAE 1010

Two distinct type of data. Taken from Paper and Mock. First ones are taken from the Paper and hoped for the best.
The latter are just mock from general

Units: SI, except pressure and hardness in MPa because Eqs. 14-15 are written in MPa.
'''

import math


# ------------------------------------------------------------------------------------------------
# Interlayer contact of the coil, Eqs. 9-17 (keys match radial_conductivity.effective_radial_conductivity)
# ------------------------------------------------------------------------------------------------

CONTACT = {
    'pressure': 8.0,              # Taken from paper: compressive stress P in MPa
    'hardness': 1133.86,          # Taken from paper: hardness H in MPa
    'roughness': 3.2215e-6,       # Taken from paper: surface roughness sigma_p in m
    'epsilon': 0.08554,           # Taken from paper: emissivity of the strip
    'oxide_thickness': 10.0e-6,   # Taken from paper: oxide layer thickness b_O in m
    'k_oxide': 2.0,               # MOCK: oxide (FeO / Fe3O4) conductivity in W/(m K), ~1-3
    'tan_theta': 0.10,            # MOCK: mean surface slope, typical ~0.05,0.15
    'eta': 1.0,                   # MOCK: emulsion correction below 800 K, 1 means no correction
}

# ------------------------------------------------------------------------------------------------
# Gas circulation and convection, Eqs. 5-8
# ------------------------------------------------------------------------------------------------

FURNACE = {
    'zeta': 1.0 / 0.909,          # Taken from paper: 1 / zeta = 0.909, for table 1 results
    'q_normal': 4.0,              # MOCK: fan flow at normal conditions in Nm3/s (14400 Nm3/h)
    'pressure': 101325.0,         # MOCK: furnace gas pressure in Pa (slight overpressure neglected)
    'psi': 1.5,                   # MOCK: vortex enhancement of Eq. 7 (fitted in the paper)
    'xi': 1.2,                    # MOCK: radiation enhancement of Eq. 8 (fitted in the paper)
    'cover_diameter': 2.10,       # MOCK: inner diameter of the inner cover in m
    'plate_gap': 0.08,            # MOCK: height of the convector plate passages in m
    'top_gap': 0.30,              # MOCK: height of the space above the top coil in m
    'channel_model': 'simple',    # 'simple': same duct on every surface, 'detailed': bore, annulus and plate slots
    # used by 'simple'
    'channel_area': 0.5,          # MOCK: flow area of the generic duct in m2
    'channel_D': 0.2,             # MOCK: hydraulic diameter of the generic duct in m
    'channel_L': 1.27,            # MOCK: flow length of the generic duct in m (about the coil width)
    # used by 'detailed'
    'cover_diameter': 2.10,
}

# ------------------------------------------------------------------------------------------------
# Data for the steel coils used. The first is the bottom layer.
# ------------------------------------------------------------------------------------------------
TABLE_3 = [  # paper case, generic steel
    {'outer_diameter': 1.869, 'width': 1.272, 'thickness': 1.50e-3, 'mass': 24490.0},
    {'outer_diameter': 1.867, 'width': 1.250, 'thickness': 1.50e-3, 'mass': 23993.0},
    {'outer_diameter': 1.837, 'width': 1.272, 'thickness': 1.48e-3, 'mass': 23557.0},
    {'outer_diameter': 1.761, 'width': 1.272, 'thickness': 1.50e-3, 'mass': 21417.0},
]


# ================================================================================================
# Ternium case
# ================================================================================================

INCH = 0.0254  # m
CUBIC_FOOT = 0.3048**3  # m3


def fahrenheit_to_kelvin(T_f: float) -> float:
    '''
    Convert a temperature from degrees Fahrenheit to K.

    Args:
        T_f: temperature in degrees F

    Returns:
        Temperature in K.
    '''
    return (T_f - 32.0) * 5.0 / 9.0 + 273.15


# ------------------------------------------------------------------------------------------------
# Data given by Ternium (original units in the comments)
# ------------------------------------------------------------------------------------------------

TERNIUM = {
    # 1. coil dimensions
    'inner_diameter': 0.508,                              # 508 mm
    'outer_diameter_max': 1.829,                          # 1,829 mm
    'width_min': 0.720,                                   # 720 mm
    'width_max': 1.320,                                   # 1,320 mm
    'thickness_min': 0.30e-3,                             # 0.30 mm
    'thickness_max': 3.49e-3,                             # 3.49 mm
    # 2. furnace program (only the set point, no time curve)
    'T_program': fahrenheit_to_kelvin(1650.0),            # 1,650 F (heating hood set point) = 1172.04 K
    # 3. grades
    'grades': ('SAE1006', 'SAE1010'),
    # 4. bell
    'bell_diameter': 2.451,                               # 2,451 mm, given as hydraulic diameter
    'operating_height': 5.250,                            # 5,250 mm, given as circulation length
    # 6. fan flow
    'flow_rate': 52000.0 * CUBIC_FOOT / 60.0,             # 52,000 ft3/min = 24.54 m3/s
    # 8-10. atmosphere and reference temperatures
    'T_gas_saturation': fahrenheit_to_kelvin(1265.0),     # 1,265 F = 958.15 K, gas not touching the coils
    'T_contact': fahrenheit_to_kelvin(1175.0),            # 1,175 F = 908.15 K, coil reference temperature
    'probe_coil': 0,                                      # 'primer rollo', taken as the bottom coil (1#)
    'probe_offset': 25.0 * INCH,                          # 25 in = 0.635 m from the inner diameter outward
}

# Not given by Ternium: 5. gas flow area, 7. roughness, 9-10. independent thermocouple readings,
# the time curve of the furnace and the number of coils per charge.

# ------------------------------------------------------------------------------------------------
# Gas circulation, Ternium case. Same correlation constants as the paper.
# ------------------------------------------------------------------------------------------------

FURNACE_TERNIUM = {
    **FURNACE,
    'flow_basis': 'actual',                       # the fan moves a fixed actual volume, not normal m3
    'q_flow': TERNIUM['flow_rate'],               # Ternium: 24.54 m3/s
    'cover_diameter': TERNIUM['bell_diameter'],   # Ternium: 2.451 m
    'channel_model': 'simple',
    'channel_D': TERNIUM['bell_diameter'],        # Ternium: hydraulic diameter 2.451 m
    'channel_L': TERNIUM['operating_height'],     # Ternium: circulation length 5.250 m
    # ESTIMATE: Ternium has no flow area (point 5). Annulus between the bell and the largest coil.
    'channel_area': math.pi / 4.0 * (TERNIUM['bell_diameter']**2 - TERNIUM['outer_diameter_max']**2),
}

# Ternium has no roughness or contact data (point 7), the paper values are kept.
# Grade-specific contact values go in properties.STEEL_GRADES[grade]['contact'].
CONTACT_TERNIUM = dict(CONTACT)

# ------------------------------------------------------------------------------------------------
# Coil stack, Ternium case. The first is the bottom coil.
# MOCK: Ternium did not give a charge. 4 coils at the largest diameter, width chosen so the stack fits
# the 5.250 m operating height with the mock plate_gap and top_gap. Edit freely within TERNIUM limits.
# ------------------------------------------------------------------------------------------------

TERNIUM_COILS = [
    {'inner_diameter': 0.508, 'outer_diameter': 1.829, 'width': 1.150, 'thickness': 1.50e-3, 'grade': 'SAE1006'},
    {'inner_diameter': 0.508, 'outer_diameter': 1.829, 'width': 1.150, 'thickness': 1.50e-3, 'grade': 'SAE1006'},
    {'inner_diameter': 0.508, 'outer_diameter': 1.829, 'width': 1.150, 'thickness': 1.50e-3, 'grade': 'SAE1010'},
    {'inner_diameter': 0.508, 'outer_diameter': 1.829, 'width': 1.150, 'thickness': 1.50e-3, 'grade': 'SAE1010'},
]

# ------------------------------------------------------------------------------------------------
# Gas temperature schedule, Ternium case
# The hold is the gas saturation temperature (1,265 F). The 1,650 F set point is the heating hood, which
# never reaches the gas, so it is not used as the gas temperature. Ramp and hold times are MOCK.
# ------------------------------------------------------------------------------------------------

HOUR = 3600.0
SCHEDULE_TERNIUM = {
    'T_start': 303.15,                        # MOCK: room temperature charge
    'T_hold': TERNIUM['T_gas_saturation'],    # Ternium: 958.15 K
    'T_end': 423.15,                          # MOCK
    'heating_time': 12.0 * HOUR,              # MOCK
    'holding_time': 10.0 * HOUR,              # MOCK
    'cooling_time': 20.0 * HOUR,              # MOCK
}


'''
Input parameters of the heat transfer model, separated by origin.

Two distinct type of data. Taken from Paper and Mock. First ones are taken from the Paper and hoped for the best.
The latter are just mock from general

Units: SI, except pressure and hardness in MPa because Eqs. 14-15 are written in MPa.
'''

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
}

# ------------------------------------------------------------------------------------------------
# Data for the steel coils used. The first is the bottom layer.
# ------------------------------------------------------------------------------------------------
TABLE_3 = [
    {'outer_diameter': 1.869, 'width': 1.272, 'thickness': 1.50e-3, 'mass': 24490.0},
    {'outer_diameter': 1.867, 'width': 1.250, 'thickness': 1.50e-3, 'mass': 23993.0},
    {'outer_diameter': 1.837, 'width': 1.272, 'thickness': 1.48e-3, 'mass': 23557.0},
    {'outer_diameter': 1.761, 'width': 1.272, 'thickness': 1.50e-3, 'mass': 21417.0},
]

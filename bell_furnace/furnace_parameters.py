'''
Input parameters of the heat transfer model, separated by origin.

PAPER values come from Yang et al. (2025). MOCK values are not given in the paper and are placeholder
estimates to make the model run; they must be replaced with Ternium data or calibrated later.
Every mock value is marked with MOCK in its comment.

Units: SI, except pressure and hardness in MPa because Eqs. 14-15 are written in MPa.
'''

# ------------------------------------------------------------------------------------------------
# Interlayer contact of the coil, Eqs. 9-17 (keys match radial_conductivity.effective_radial_conductivity)
# ------------------------------------------------------------------------------------------------

CONTACT = {
    'pressure': 8.0,              # PAPER: compressive stress P in MPa
    'hardness': 1133.86,          # PAPER: hardness H in MPa
    'roughness': 3.2215e-6,       # PAPER: surface roughness sigma_p in m
    'epsilon': 0.08554,           # PAPER: emissivity of the strip
    'oxide_thickness': 10.0e-6,   # PAPER: oxide layer thickness b_O in m
    'k_oxide': 2.0,               # MOCK: oxide (FeO / Fe3O4) conductivity in W/(m K), typical 1 to 3
    'tan_theta': 0.10,            # MOCK: mean surface slope, typical 0.05 to 0.15 for sigma_p near 3 um
    'eta': 1.0,                   # MOCK: emulsion correction below 800 K, 1 means no correction
}

# ------------------------------------------------------------------------------------------------
# Gas circulation and convection, Eqs. 5-8
# ------------------------------------------------------------------------------------------------

FURNACE = {
    'zeta': 1.0 / 0.909,          # PAPER: 1 / zeta = 0.909, reproduces Table 1 (fd = 0.202, 0.183, 0.167)
    'q_normal': 4.0,              # MOCK: fan flow at normal conditions in Nm3/s (14400 Nm3/h)
    'pressure': 101325.0,         # MOCK: furnace gas pressure in Pa (slight overpressure neglected)
    'psi': 1.5,                   # MOCK: vortex enhancement of Eq. 7 (fitted in the paper, not reported)
    'xi': 1.2,                    # MOCK: radiation enhancement of Eq. 8 (fitted in the paper, not reported)
    # Channel model: 'simple' gives every surface the same duct (convection.simple_channel_geometry),
    # 'detailed' builds bore, annulus and plate slots from the furnace dimensions (convection.channel_geometry)
    'channel_model': 'simple',
    # used by 'simple'
    'channel_area': 0.5,          # MOCK: flow area of the generic duct in m2
    'channel_D': 0.2,             # MOCK: hydraulic diameter of the generic duct in m
    'channel_L': 1.27,            # MOCK: flow length of the generic duct in m (about the coil width)
    # used by 'detailed'
    'cover_diameter': 2.10,       # MOCK: inner diameter of the inner cover in m
    'plate_gap': 0.08,            # MOCK: height of the convector plate passages in m
    'top_gap': 0.30,              # MOCK: height of the space above the top coil in m
}

# ------------------------------------------------------------------------------------------------
# Coils of the buried thermocouple test, Table 3, bottom coil (1#) first
# ------------------------------------------------------------------------------------------------

TABLE_3 = [
    {'outer_diameter': 1.869, 'width': 1.272, 'thickness': 1.50e-3, 'mass': 24490.0},
    {'outer_diameter': 1.867, 'width': 1.250, 'thickness': 1.50e-3, 'mass': 23993.0},
    {'outer_diameter': 1.837, 'width': 1.272, 'thickness': 1.48e-3, 'mass': 23557.0},
    {'outer_diameter': 1.761, 'width': 1.272, 'thickness': 1.50e-3, 'mass': 21417.0},
]

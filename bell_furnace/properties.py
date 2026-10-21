'''
Temperature-dependent properties of the protective gas (pure hydrogen) and the steel strip.

All inputs and outputs are NumPy arrays in SI units, with temperatures in K.

Sources:
    hydrogen: data/hydrogen.csv, generated with CoolProp by scripts/generate_hydrogen_table.py
    steel:    EN 1993-1-2 (Eurocode 3), section 3.4.1, carbon steel, 20 to 1200 C
'''

from pathlib import Path

import numpy as np

# Load data of values 
_DATA = Path(__file__).resolve().parent / 'data'
_H2_TABLE = np.loadtxt(_DATA / 'hydrogen.csv', delimiter=',')
_STEEL_TABLE = np.loadtxt(_DATA / 'steel.csv', delimiter=',')


def hydrogen_properties(T_g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Interpolate properties (lineraly) of pure hydrogen from the CoolProp table.

    The table is at 1 atm. Viscosity, conductivity and specific heat are practically independent
    of pressure near atmospheric. We ignore pressure effects on density since it's a small error.

    Args:
        T_g: gas temperature in K

    Returns:
        Tuple (rho, mu, k, cp) of arrays with the shape of T_g: density in kg/m3, dynamic
        viscosity in Pa s, thermal conductivity in W/(m K) and specific heat in J/(kg K).

    Raises:
        ValueError if any temperature is outside the table range.
    '''
    T_g = np.asarray(T_g, dtype=float)
    T_table = _H2_TABLE[:, 0]
    if np.any(T_g < T_table[0]) or np.any(T_g > T_table[-1]):
        raise ValueError(f'temperature must be between {T_table[0]:.0f} and {T_table[-1]:.0f} K')

    rho = np.interp(T_g, T_table, _H2_TABLE[:, 1])
    mu = np.interp(T_g, T_table, _H2_TABLE[:, 2])
    k = np.interp(T_g, T_table, _H2_TABLE[:, 3])
    cp = np.interp(T_g, T_table, _H2_TABLE[:, 4])

    return rho, mu, k, cp


 
def steel_properties(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Interpolate properties of the steel strip from the steel table.
 
    Density is constant in the current table, so thermal expansion is neglected. The thermal
    conductivity is that of solid steel and is used as the axial conductivity.
 
    Args:
        T: steel temperature in K
 
    Returns:
        Tuple (rho, cp, k) of arrays with the shape of T: density in kg/m3, specific heat in
        J/(kg K) and thermal conductivity in W/(m K).
 
    Raises:
        ValueError if any temperature is outside the table range.
    '''
    T = np.asarray(T, dtype=float)
    T_table = _STEEL_TABLE[:, 0]
    if np.any(T < T_table[0]) or np.any(T > T_table[-1]):
        raise ValueError(f'temperature must be between {T_table[0]:.2f} and {T_table[-1]:.2f} K')
 
    rho = np.interp(T, T_table, _STEEL_TABLE[:, 1])
    cp = np.interp(T, T_table, _STEEL_TABLE[:, 2])
    k = np.interp(T, T_table, _STEEL_TABLE[:, 3])
 
    return rho, cp, k
 
'''
Generate the hydrogen property table used by properties.hydrogen_properties.

Queries CoolProp for normal hydrogen at a fixed pressure over a temperature grid and writes
data/hydrogen.csv with columns T, rho, mu, k, cp. Run once; the main code only reads the CSV
and does not need CoolProp installed.

CoolProp hydrogen sources:
    equation of state:    Leachman et al. (2009), J. Phys. Chem. Ref. Data 38, 721
    viscosity:            Muzny et al. (2013), J. Chem. Eng. Data 58, 969
    thermal conductivity: Assael et al. (2011), J. Phys. Chem. Ref. Data 40, 033101

Usage:
    python scripts/generate_hydrogen_table.py
'''

from pathlib import Path

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np



def generate_hydrogen_table(*, output: Path, T_min: float = 250.0, T_max: float = 1200.0, step: float = 5.0, pressure: float = 101325.0) -> np.ndarray:
    '''
    Compute hydrogen properties with CoolProp and save them to a CSV file.

    Args:
        output:   path of the CSV file to write
        T_min:    lowest temperature in K
        T_max:    highest temperature in K
        step:     temperature spacing in K
        pressure: absolute pressure in Pa

    Returns:
        Array of shape (n, 5) with columns T in K, rho in kg/m3, mu in Pa s, k in W/(m K)
        and cp in J/(kg K).

    Raises:
        ValueError if the temperature range, step or pressure is invalid.
    '''
    if T_min <= 0.0 or T_max <= T_min:
        raise ValueError('temperature range must satisfy 0 < T_min < T_max')
    if step <= 0.0:
        raise ValueError('step must be positive in K')
    if pressure <= 0.0:
        raise ValueError('pressure must be positive in Pa')

    T = np.arange(T_min, T_max + 0.5 * step, step)
    table = np.empty((T.size, 5))
    table[:, 0] = T
    for column, key in enumerate(['D', 'V', 'L', 'C'], start=1):
        table[:, column] = [CP.PropsSI(key, 'T', t, 'P', pressure, 'Hydrogen') for t in T]

    header = (
        f'Normal hydrogen at {pressure:.0f} Pa, generated with CoolProp {CoolProp.__version__}\n'
        'T [K], rho [kg/m3], mu [Pa s], k [W/(m K)], cp [J/(kg K)]'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, table, delimiter=',', header=header, fmt=['%.1f', '%.6e', '%.6e', '%.6e', '%.6e'])
    return table


if __name__ == '__main__':
    path = Path(__file__).resolve().parent.parent / 'data' / 'hydrogen.csv'
    result = generate_hydrogen_table(output=path)
    print(f'wrote {result.shape[0]} rows to {path}')
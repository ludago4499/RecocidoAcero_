'''
Process the Ternium production data (Recocido -Tec.xlsx) and rebuild the charges.

Shared by the three branches: the physical model reads the coils of each charge, the Random Forest
reads the charge features and the optimizer reads both. The Excel is confidential and is not in the
repository (see .gitignore): put it in the project root on your computer.

The Excel has one row per (coil, lab sample), so a coil can repeat. This module gives:
    coils:          one row per coil, SI units, outer diameter derived from mass and width
    loads:          one row per charge (RH- Carga), SI units, process times, flags and lab means
    load_features:  loads in the units and names of analisis_fase2.py, for PCA and the Random Forest
    position_features: one column per coil position (mass, width, thickness of coil 1 to 5)

Units in the Excel (row 0 of the sheet):
    widths and thicknesses in mm, masses in kg, length in m
    RH- Temperatura Saturacion in F, RH- Tiempo Saturacion, RH-Tiempo Neto carga and RH-Tiempo STD Carga in min
    RH- Posicion: 1 is the bottom coil

Every unit conversion and every derived column is done here, so the rest of the code only sees SI.

Usage, from the project root:
    python -m pipeline.data_processing                    summary of the cleaned data
'''

from pathlib import Path

import numpy as np
import pandas as pd

from bell_furnace.furnace_parameters import TERNIUM, fahrenheit_to_kelvin
from bell_furnace.properties import FAMILY_TO_GRADE, steel_density

ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT / 'Recocido -Tec.xlsx'
MINUTE = 60.0  # s

# Excel column -> short name used in the code
COIL_COLUMNS = {
    'Material': 'coil_id',
    'RH- Carga': 'load_id',
    'RH- Posicion': 'position',
    'RH- Ancho': 'width_mm',
    'RH- Espesor': 'thickness_mm',
    'RH- Peso Entrada': 'mass',
    'RH- Largo': 'length',
    'RH-Familia acero': 'family',
    'RH-Acero': 'steel_code',
    'RH- Peso de Salida': 'mass_out',
}
LOAD_COLUMNS = {
    'RH- Carga': 'load_id',
    'RH- Horno': 'furnace_id',
    'RH- Base': 'base_id',
    'RH- Enfriador': 'cooler_id',
    'RH- Peso Carga': 'load_mass',
    'RH- Temperatura Saturacion': 'T_sat_F',
    'RH- Tiempo Saturacion': 't_sat_min',
    'RH-Tiempo Neto carga': 't_net_min',
    'RH-Tiempo STD Carga': 't_std_min',
    'RH-Eficiencia': 'efficiency',
    'RH- Fecha Produccion': 'date',
}

# lab tests, averaged per charge (a coil can have several samples)
LAB_COLUMNS = {
    'Tensión L - Elongación': 'lab_elongation',
    'Tensión L - Punto de cedencia': 'lab_yield',
    'Tensión L - Resistencia': 'lab_tensile',
    'Elongación Temple': 'temper_elongation',
}

# Plausibility limits used to flag rows, not to drop them
DENSITY_RANGE = (7000.0, 8500.0)  # kg/m3, apparent density from mass / (width thickness length)
T_SAT_MIN_F = 1000.0              # F, below this the saturation temperature is taken as missing (0 in 35 % of loads)
T_NET_MAX_MIN = 4500.0            # min, net times above this (75 h) are taken as outliers


def read_raw(*, path: Path = EXCEL_PATH) -> pd.DataFrame:
    '''
    Read the Excel sheet as given by Ternium.

    Args:
        path:   path of Recocido -Tec.xlsx

    Returns:
        Data frame with one row per Excel row and stripped column names.

    Raises:
        FileNotFoundError if the file does not exist.
    '''
    if not Path(path).exists():
        raise FileNotFoundError(f'{path} not found')
    df = pd.read_excel(path, sheet_name=0, header=1, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]
    return df


def outer_diameter_from_mass(*, mass: np.ndarray, width: np.ndarray, inner_diameter: float, density: np.ndarray) -> np.ndarray:
    '''
    Compute the outer diameter of a coil from its mass, assuming a solid ring of steel.

    Ternium does not give the outer diameter per coil. The strip length is not used because it has
    bad values (apparent densities up to 4e5 kg/m3); mass and width are reliable.

    Args:
        mass:           coil mass in kg
        width:          coil width in m
        inner_diameter: inner diameter in m
        density:        steel density in kg/m3

    Returns:
        Outer diameter in m.
    '''
    return np.sqrt(inner_diameter**2 + 4.0 * mass / (np.pi * density * width))


def build_coils(*, raw: pd.DataFrame, family_to_grade: dict[str, str] = FAMILY_TO_GRADE) -> pd.DataFrame:
    '''
    Build the coil table: one row per coil, SI units.

    Args:
        raw:             data frame from read_raw
        family_to_grade: steel family (RH-Familia acero) -> key of properties.STEEL_GRADES

    Returns:
        Data frame with columns coil_id, load_id, position, width (m), thickness (m), mass (kg),
        family, steel_code, grade, density (kg/m3), inner_diameter (m), outer_diameter (m),
        apparent_density (kg/m3) and geometry_ok (apparent density inside DENSITY_RANGE).

    Raises:
        ValueError if a steel family has no entry in family_to_grade.
    '''
    coils = raw.drop_duplicates(subset=['Material'])[list(COIL_COLUMNS)].rename(columns=COIL_COLUMNS).copy()

    unknown = set(coils['family'].dropna()) - family_to_grade.keys()
    if unknown:
        raise ValueError(f'steel families without a grade: {sorted(unknown)}, add them to properties.FAMILY_TO_GRADE')

    coils['width'] = coils.pop('width_mm') / 1000.0
    coils['thickness'] = coils.pop('thickness_mm') / 1000.0
    coils['grade'] = coils['family'].map(family_to_grade)
    coils['density'] = coils['grade'].map(steel_density)
    coils['inner_diameter'] = TERNIUM['inner_diameter']
    coils['outer_diameter'] = outer_diameter_from_mass(mass=coils['mass'].to_numpy(), width=coils['width'].to_numpy(), inner_diameter=TERNIUM['inner_diameter'], density=coils['density'].to_numpy())
    coils['apparent_density'] = coils['mass'] / (coils['width'] * coils['thickness'] * coils['length'])
    coils['geometry_ok'] = coils['apparent_density'].between(*DENSITY_RANGE)
    return coils.sort_values(['load_id', 'position']).reset_index(drop=True)


def build_loads(*, raw: pd.DataFrame, coils: pd.DataFrame) -> pd.DataFrame:
    '''
    Build the load table: one row per charge, SI units, with flags instead of dropped rows.

    Args:
        raw:   data frame from read_raw
        coils: data frame from build_coils

    Returns:
        Data frame indexed by load_id with columns furnace_id, base_id, cooler_id, load_mass (kg),
        T_sat (K, nan when missing), t_sat, t_net, t_std (s), efficiency, date, n_coils,
        stack_width (m), thickness_mean, thickness_max (m), width_max (m), dominant_family,
        coil_mass_sum (kg) and the flags T_sat_ok, t_sat_ok, t_net_ok, geometry_ok and valid
        (all flags true).
    '''
    loads = raw.drop_duplicates(subset=['RH- Carga'])[list(LOAD_COLUMNS)].rename(columns=LOAD_COLUMNS).set_index('load_id')

    T_ok = loads['T_sat_F'] >= T_SAT_MIN_F
    loads['T_sat'] = np.where(T_ok, fahrenheit_to_kelvin(loads['T_sat_F']), np.nan)
    loads['t_sat'] = loads.pop('t_sat_min') * MINUTE
    loads['t_net'] = loads.pop('t_net_min') * MINUTE
    loads['t_std'] = loads.pop('t_std_min') * MINUTE

    grouped = coils.groupby('load_id')
    loads = loads.join(grouped.agg(
        n_coils=('coil_id', 'count'),
        stack_width=('width', 'sum'),
        thickness_mean=('thickness', 'mean'),
        thickness_max=('thickness', 'max'),
        width_max=('width', 'max'),
        dominant_family=('family', lambda s: s.mode().iloc[0]),
        coil_mass_sum=('mass', 'sum'),
        mass_out_sum=('mass_out', 'sum'),
        geometry_ok=('geometry_ok', 'all'),
    ))
    lab = raw[['RH- Carga', *LAB_COLUMNS]].rename(columns={'RH- Carga': 'load_id', **LAB_COLUMNS})
    loads = loads.join(lab.groupby('load_id').mean(numeric_only=True))

    loads['T_sat_ok'] = T_ok
    loads['t_sat_ok'] = loads['t_sat'] > 0.0
    loads['t_net_ok'] = loads['t_net'] < T_NET_MAX_MIN * MINUTE
    loads['valid'] = loads[['T_sat_ok', 't_sat_ok', 't_net_ok', 'geometry_ok']].all(axis=1)
    return loads


def load_data(*, path: Path = EXCEL_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Read the Excel and build both tables.

    Args:
        path:   path of Recocido -Tec.xlsx

    Returns:
        Tuple (coils, loads), see build_coils and build_loads.
    '''
    raw = read_raw(path=path)
    coils = build_coils(raw=raw)
    loads = build_loads(raw=raw, coils=coils)
    return coils, loads


def load_coils(*, coils: pd.DataFrame, load_id: int) -> list[dict]:
    '''
    Return the coils of one charge in the format of cases.build_stack_case, bottom coil first.

    Args:
        coils:   data frame from build_coils
        load_id: value of RH- Carga

    Returns:
        List of coil dicts with keys coil_id, inner_diameter, outer_diameter, width, thickness
        (m), grade, family and mass (kg).

    Raises:
        KeyError if the charge is not in the table.
    '''
    rows = coils[coils['load_id'] == load_id].sort_values('position')
    if rows.empty:
        raise KeyError(f'load {load_id} not found')
    keys = ['coil_id', 'inner_diameter', 'outer_diameter', 'width', 'thickness', 'grade', 'family', 'mass']
    return [{k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()} for row in rows[keys].to_dict('records')]


def load_features(*, loads: pd.DataFrame, only_valid: bool = True) -> pd.DataFrame:
    '''
    Return the charge features with the names and units used in analisis_fase2.py and PCA.ipynb.

    Same cleaning as analisis_fase2.py (T_sat >= 1000 F, t_sat > 0, t_net < 4500 min, coil
    geometry consistent), so the Random Forest and the PCA see the same 520 charges.

    Args:
        loads:      data frame from build_loads
        only_valid: true keeps only the charges with every flag true

    Returns:
        Data frame indexed by load_id with columns Peso_Carga_t, num_rollos, espesor_max,
        espesor_mean (mm), ancho_max (mm), Temp_Sat_C, Tiempo_Sat_h, Tiempo_Neto_h, Tiempo_STD_h,
        familia_dom, Horno, Base, Enfriador, the lab means and valid.
    '''
    data = loads[loads['valid']] if only_valid else loads
    features = pd.DataFrame({
        'Peso_Carga_t': data['load_mass'] / 1000.0,
        'num_rollos': data['n_coils'],
        'espesor_max': data['thickness_max'] * 1000.0,
        'espesor_mean': data['thickness_mean'] * 1000.0,
        'ancho_max': data['width_max'] * 1000.0,
        'Temp_Sat_C': data['T_sat'] - 273.15,
        'Tiempo_Sat_h': data['t_sat'] / 3600.0,
        'Tiempo_Neto_h': data['t_net'] / 3600.0,
        'Tiempo_STD_h': data['t_std'] / 3600.0,
        'familia_dom': data['dominant_family'],
        'Horno': data['furnace_id'],
        'Base': data['base_id'],
        'Enfriador': data['cooler_id'],
        'valid': data['valid'],
    })
    return features.join(data[list(LAB_COLUMNS.values())])


def position_features(*, coils: pd.DataFrame, max_position: int = 5) -> pd.DataFrame:
    '''
    Return one column per coil position, for models that use the order of the stack.

    Charges with fewer coils than max_position get 0 in the empty positions.

    Args:
        coils:        data frame from build_coils
        max_position: highest position kept (1 is the bottom coil)

    Returns:
        Data frame indexed by load_id with columns mass_p1..p5 (t), width_p1..p5 (mm) and
        thickness_p1..p5 (mm).
    '''
    data = coils[coils['position'] <= max_position].copy()
    data['mass_t'] = data['mass'] / 1000.0
    data['width_mm'] = data['width'] * 1000.0
    data['thickness_mm'] = data['thickness'] * 1000.0
    wide = data.pivot_table(index='load_id', columns='position', values=['mass_t', 'width_mm', 'thickness_mm'], aggfunc='first').fillna(0.0)
    names = {'mass_t': 'mass', 'width_mm': 'width', 'thickness_mm': 'thickness'}
    wide.columns = [f'{names[v]}_p{int(p)}' for v, p in wide.columns]
    return wide


if __name__ == '__main__':
    coils, loads = load_data()
    print(f'{len(coils)} coils in {len(loads)} loads')
    print(f"coils per load: {loads['n_coils'].value_counts().sort_index().to_dict()}")
    print(f"families: {coils['family'].value_counts().to_dict()}")
    print(f"grades used by the model: {coils['grade'].value_counts().to_dict()}")
    print(f"outer diameter: {coils['outer_diameter'].min():.3f} to {coils['outer_diameter'].max():.3f} m")
    print('flags (share of loads that pass):')
    for flag in ['T_sat_ok', 't_sat_ok', 't_net_ok', 'geometry_ok', 'valid']:
        print(f'  {flag:12s} {loads[flag].mean() * 100:5.1f} %')
    valid = loads[loads['valid']]
    print(f"features for PCA / Random Forest: {load_features(loads=loads).shape}, by position: {position_features(coils=coils).shape}")
    print(f"valid loads: T_sat {valid['T_sat'].median() - 273.15:.0f} C, t_sat {valid['t_sat'].median() / 3600:.1f} h, t_net {valid['t_net'].median() / 3600:.1f} h (medians)")

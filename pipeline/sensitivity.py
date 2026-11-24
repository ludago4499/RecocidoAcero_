'''
Sensitivity of the thermal behaviour to the model parameters and to the charge.

Owner: feature/physical-model. One-at-a-time variations around a base charge, run with
bell_furnace.cases.build_stack_case and bell_furnace.metrics. Two uses:
    calibration: which uncertain parameters (psi, xi, channel_area, contact) move the heating time
                 enough to close the gap with the plant data
    optimization: which charge variables (mass, thickness, number of coils, T_sat) matter physically
'''

import pandas as pd

# Parameters to vary: name -> (dict that holds it, key)
PARAMETERS = {
    'psi': ('furnace', 'psi'),
    'xi': ('furnace', 'xi'),
    'channel_area': ('furnace', 'channel_area'),
    'q_flow': ('furnace', 'q_flow'),
    'roughness': ('contact', 'roughness'),
    'pressure': ('contact', 'pressure'),
    'k_oxide': ('contact', 'k_oxide'),
}


def one_at_a_time(*, case: dict, parameters: list[str], factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    '''
    Rerun a case multiplying one parameter at a time by each factor.

    Args:
        case:       base case dict from bell_furnace.cases
        parameters: keys of PARAMETERS to vary
        factors:    multipliers of the base value

    Returns:
        Data frame with columns parameter, factor, value and the metrics of stack_metrics (h).

    Raises:
        NotImplementedError until the physical-model branch implements it.
    '''
    raise NotImplementedError('feature/physical-model')

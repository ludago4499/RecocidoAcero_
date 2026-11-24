'''
Train, validate and use the Random Forest that predicts the net annealing time of a charge.

Owner: feature/random-forest. Start from the Random Forest part of analisis_fase2.py, with the
features of data_processing.load_features (same names and the same 520 charges). Validate on charges
not used for training (hold-out or k-fold, grouped by date if possible).

The interface below is fixed so the optimizer can use the model before it is finished; the bodies
are for the owner to write.
'''

import numpy as np
import pandas as pd

# Columns of data_processing.load_features used as inputs; categorical ones are one-hot encoded
FEATURES = ['Peso_Carga_t', 'num_rollos', 'espesor_max', 'espesor_mean', 'ancho_max', 'Temp_Sat_C', 'Tiempo_Sat_h']
CATEGORICAL = ['familia_dom', 'Horno', 'Enfriador']
TARGET = 'Tiempo_Neto_h'


def train(*, features: pd.DataFrame, seed: int = 42) -> dict:
    '''
    Train and validate the Random Forest.

    Args:
        features: data frame from data_processing.load_features
        seed:     random seed

    Returns:
        Dict with model (fitted regressor), columns (input columns after encoding), metrics
        (validation MAE, RMSE and R2 in h) and importances (pd.Series).

    Raises:
        NotImplementedError until the random-forest branch implements it.
    '''
    raise NotImplementedError('feature/random-forest')


def predict(*, model: dict, features: pd.DataFrame) -> np.ndarray:
    '''
    Predict the net annealing time of charges.

    Args:
        model:    dict returned by train
        features: data frame with the columns FEATURES and CATEGORICAL, one row per charge

    Returns:
        Predicted net time in h, one value per row.

    Raises:
        NotImplementedError until the random-forest branch implements it.
    '''
    raise NotImplementedError('feature/random-forest')


def candidate_features(*, load: dict) -> pd.DataFrame:
    '''
    Convert a candidate charge of the optimizer (see constraints) into one row of features.

    Args:
        load:   candidate charge dict with coils, T_sat, t_sat and the equipment ids

    Returns:
        Data frame with one row and the columns FEATURES and CATEGORICAL, in the units of
        data_processing.load_features.

    Raises:
        NotImplementedError until the random-forest branch implements it.
    '''
    raise NotImplementedError('feature/random-forest')

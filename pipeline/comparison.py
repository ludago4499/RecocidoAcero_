'''
Compare the physical, predictive and optimization results.

Integration step, after the three branches are merged. Inputs:
    physical:      results/ternium_loads/model_vs_data.csv (scripts/run_ternium_loads.py)
    predictive:    random_forest.predict on the same charges
    optimization:  candidates of simulated_annealing, checked with the physical model

The time of each method is not the same quantity (model heating vs plant net time), so the comparison
is by trend and ranking, not one to one.
'''

import pandas as pd


def compare_predictions(*, physical: pd.DataFrame, predicted: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    '''
    Join the physical model, the Random Forest and the plant data per charge.

    Args:
        physical:  model_vs_data.csv as a data frame
        predicted: Random Forest net time in h, indexed by load_id
        features:  data frame from data_processing.load_features

    Returns:
        Data frame indexed by load_id with the plant times, the Random Forest prediction, the model
        heating time and their errors.

    Raises:
        NotImplementedError until implemented at integration.
    '''
    raise NotImplementedError('integration')


def compare_optimum(*, baseline: dict, optimum: dict) -> pd.DataFrame:
    '''
    Compare an optimized charge against a real one, with both evaluators and the constraints.

    Args:
        baseline: candidate charge dict of a real charge
        optimum:  candidate charge dict returned by simulated_annealing.optimize

    Returns:
        Data frame with one row per charge: Random Forest time, physical heating time, constraint
        violations.

    Raises:
        NotImplementedError until implemented at integration.
    '''
    raise NotImplementedError('integration')

'''
Define what the optimizer minimizes: J(x) for a candidate charge x.

Owner: feature/simulated-annealing. The final objective is not fixed yet (see the architecture
document): minimizing time alone does not guarantee a correct annealing, so the objective always
adds the constraint penalties.

Two evaluators, so the two optimizations can be compared:
    'random_forest': fast, predicted net time from random_forest.predict
    'physical':      slow, cold spot heating time from bell_furnace.metrics plus the soak time
'''

EVALUATORS = ('random_forest', 'physical')


def objective(*, load: dict, evaluator: str = 'random_forest', model: dict | None = None, penalty_weight: float = 1.0e3) -> float:
    '''
    Evaluate J(x) for one candidate charge; lower is better.

    Args:
        load:           candidate charge dict (see constraints)
        evaluator:      'random_forest' or 'physical'
        model:          dict from random_forest.train, needed with 'random_forest'
        penalty_weight: penalty per violated constraint (constraints.penalty), in h

    Returns:
        Objective value in h (time per charge, or per tonne if the team decides so, plus penalties).

    Raises:
        NotImplementedError until the simulated-annealing branch implements it.
    '''
    raise NotImplementedError('feature/simulated-annealing')

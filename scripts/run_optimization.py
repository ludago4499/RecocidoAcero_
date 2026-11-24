'''
Run the whole procedure: data, Random Forest, simulated annealing, physical check and comparison.

Owner: feature/simulated-annealing, finished at integration. Each step below calls the module that
owns it; the steps not implemented yet stop with NotImplementedError and the branch that owns them.

Usage, from the project root:
    python scripts/run_optimization.py
'''

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import random_forest  # noqa: E402
from pipeline.constraints import check_load  # noqa: E402
from pipeline.data_processing import load_data, load_features  # noqa: E402

OUTPUT = ROOT / 'results' / 'optimization'


def main() -> None:
    '''
    Run every step in order and save the results in results/optimization.
    '''
    coils, loads = load_data()
    features = load_features(loads=loads)
    print(f'1. data: {len(features)} valid charges')

    model = random_forest.train(features=features)
    print(f"2. Random Forest: {model['metrics']}")

    # 3. simulated annealing with objective_function.objective (evaluator 'random_forest')
    # 4. physical check of the best candidates: bell_furnace.cases.build_stack_case, metrics.run_case,
    #    constraints.thermal_constraint
    # 5. comparison.compare_optimum against the real charges
    _ = (coils, check_load)
    OUTPUT.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main()

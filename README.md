# RecocidoAcero_
Optimización del Tiempo de Recocido en Aceros Industriales Usando Modelado Predictivo

## Datos de Ternium (confidenciales)

`Recocido -Tec.xlsx` **no se sube al repositorio** (está en `.gitignore`, junto con las tablas derivadas en `results/ternium_loads/`, `results/random_forest/` y `results/optimization/`). Cada integrante coloca su copia en la raíz del proyecto. Antes de hacer commit de un notebook, borra las salidas que muestren filas del Excel (Kernel > Restart & Clear Output).

## Estructura

| Carpeta / archivo | Función | Rama dueña |
|---|---|---|
| `bell_furnace/` | Modelo físico: malla, propiedades, convección, solver, casos y métricas | `feature/physical-model` |
| `pipeline/data_processing.py` | Limpieza del Excel, cargas, variables para PCA y Random Forest | `main` (compartido) |
| `pipeline/constraints.py` | Restricciones geométricas, operativas y térmicas | `main` (compartido) |
| `pipeline/random_forest.py` | Entrenamiento, validación y predicción | `feature/random-forest` |
| `pipeline/objective_function.py` | Función objetivo J(x) | `feature/simulated-annealing` |
| `pipeline/simulated_annealing.py` | Recocido simulado | `feature/simulated-annealing` |
| `pipeline/sensitivity.py` | Sensibilidad y calibración del modelo físico | `feature/physical-model` |
| `pipeline/comparison.py` | Comparación física, predictiva y de optimización | integración |
| `scripts/run_optimization.py` | Procedimiento completo | `feature/simulated-annealing` |
| `scripts/run_ternium_loads.py` | Simula las cargas reales y escribe modelo vs datos | `feature/physical-model` |

Las firmas de las funciones en `pipeline/` ya están fijadas; cada rama implementa el cuerpo de sus archivos y no modifica los de otras ramas. Los cambios a `data_processing.py` o `constraints.py` van a `main` y se avisan al equipo.

## Uso

```
pip install -r requirements.txt
python -m pipeline.data_processing          # resumen de los datos limpios
python scripts/run_ternium_loads.py         # modelo físico en las 520 cargas válidas
python -m pytest tests
```

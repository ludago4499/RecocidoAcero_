# Verificación del recocido simulado

Etapa 3 de `feature/simulated-annealing`. Se verificó el minimizador con dos funciones continuas y una permutación pequeña cuyo espacio completo se puede enumerar. Los valores siguientes son costos matemáticos sintéticos. La reducción del tiempo neto industrial queda pendiente de evaluación.

## Punto de partida

El clon se abrió limpio en `d51e4b3`, la punta remota de las tres ramas de desarrollo en la inspección. `origin/main` estaba dos commits por delante, en `ad73364`. La diferencia entre los árboles se limita al reemplazo de `analisis_fase2.py` por `limpiezadatos.py` con cambios en la limpieza. El trabajo se hizo sobre la rama del optimizador.

| Interfaz inspeccionada | Estado al iniciar |
|---|---|
| `pipeline/simulated_annealing.py` | `optimize` y `neighbor_swap_coils` declaraban las firmas y lanzaban `NotImplementedError`. |
| `pipeline/objective_function.py` | `objective(*, load, evaluator, model, penalty_weight)` declaraba los evaluadores y la penalización. El cuerpo estaba pendiente. |
| `pipeline/constraints.py` | `check_load` y `thermal_constraint` devuelven listas de problemas. `penalty` las transforma en un costo. |
| `pipeline/data_processing.py` | Reconstruye rollos, cargas y características desde el Excel privado. |
| `pipeline/random_forest.py` | Las firmas de entrenamiento, construcción de características y predicción estaban definidas. Los cuerpos estaban pendientes. |
| `scripts/run_optimization.py` | Secuencia de integración que comienza por cargar el Excel y entrenar el modelo. |

## Implementación

`optimize` conserva los argumentos por nombre y los campos públicos de retorno. Acepta estados que soporten copia profunda. La configuración inicial debe ser factible y tener costo escalar real finito. Cada vecino recibe una copia del estado vigente y el generador de NumPy de la corrida. La función objetivo y el predicado opcional `feasible` también reciben copias independientes. Un vecino puede, por ejemplo, permutar una lista en sitio sin alterar el mejor estado guardado.

Para cada propuesta se calcula $\Delta=J(x')-J(x)$. Si $\Delta\leq0$ se acepta. Para $\Delta>0$ se acepta si un número uniforme en $[0,1)$ es menor que $\exp(-\Delta/T_k)$. La regla convencional presupone propuestas simétricas para la interpretación de Metropolis. Los vecinos de los tres ensayos son simétricos. La temperatura **algorítmica** tiene unidades de costo y se usa al decidir un movimiento. En un futuro evaluador que devuelva horas, su escala sería de horas, no de grados del horno.

La propuesta $k=1,\ldots,N$ usa $T_k=T_0\alpha^{k-1}$. El historial registra la temperatura usada, la aceptación, el estado de evaluación, el costo vigente y el mejor costo **después** de la decisión. La tasa acumulada incluye todas las propuestas, también las rechazadas por restricciones. La temperatura continúa descendiendo en cada intento. Si llega a cero por subdesbordamiento numérico, solo se aceptan empates y mejoras.

- Un candidato inviable se rechaza antes de llamar al objetivo. Un costo NaN o infinito también se rechaza y cuenta como una evaluación. Un error del evaluador se propaga para que pueda diagnosticarse.
- `x_current` y `x_best` son independientes. `n_evaluations` cuenta llamadas reales, incluida la inicial. La corrida es reproducible si el objetivo y la factibilidad son deterministas y el vecino usa el generador recibido.
- El ejemplo público `neighbor_swap_coils` intercambia registros completos entre la carga y el conjunto disponible. Mantiene las propiedades intrínsecas y la cantidad de rollos. La factibilidad de una carga real requerirá comprobaciones adicionales.

## Protocolo

Se realizaron 20 corridas por problema y método con semillas enteras de 0 a 19. Cada corrida tuvo exactamente **2 001 llamadas al objetivo**, contadas durante la ejecución. Se mantuvieron el dominio y el punto inicial comunes. Los dos métodos de recocido usaron la misma distribución de vecinos. La búsqueda aleatoria muestreó uniformemente en el mismo dominio. La calibración y el muestreo aleatorio usaron secuencias separadas del generador de movimientos.

| Método | Uso de las 2 001 evaluaciones |
|---|---|
| Convencional | Una inicial y 2 000 propuestas. Se conservaron los valores públicos de referencia $T_0=1$, $\alpha=0.95$. |
| Calibrado | 64 de piloto, una inicial y 1 936 propuestas. El piloto obtiene 32 pares independientes de ancla uniforme y vecino. Sus estados se usan solo para estimar temperatura. |
| Aleatorio | Una inicial y 2 000 muestras uniformes independientes, con reposición. |

Para la calibración se toma la mediana $d_+$ de los incrementos positivos finitos. Se elige $T_0=-d_+/\log(0.8)$, de modo que un incremento de ese tamaño tenga probabilidad inicial de aceptación 0.8. La tasa observada también depende de los demás incrementos y de las mejoras. Si el piloto carece de incrementos positivos finitos, se informa el error. Para que la **última propuesta** use $T_f=0.001T_0$, se calcula $\alpha=\exp[(\log T_f-\log T_0)/(N-1)]$.

Sphere usa $J(x)=\sum_i x_i^2$ y Rastrigin usa $J(x)=\sum_i[x_i^2+10(1-\cos(2\pi x_i))]$. Ambas son bidimensionales, con mínimo cero en el origen, dominio $[-5.12,5.12]^2$ e inicio $(4.2,-3.8)$. El vecino suma ruido normal independiente de desviación estándar 0.55 por coordenada y refleja los pasos que salen del dominio.

La ruta usa ocho nodos, depósito fijo en 0 y costos dirigidos enteros. Se parte del orden ascendente y cada vecino intercambia dos posiciones distintas. Una enumeración separada de las $7!=5\,040$ permutaciones, incluyendo el regreso al depósito, dio el mínimo **54**. Una ruta que lo alcanza es **0 → 7 → 2 → 4 → 6 → 1 → 3 → 5 → 0**. La enumeración se usa para la referencia y queda fuera del presupuesto de los tres buscadores.

## Resultados

Cada celda muestra la **mediana** del mejor costo final y, entre corchetes, el intervalo mínimo–máximo entre semillas. Un costo menor es mejor.

| Problema | Óptimo | Convencional | Calibrado | Aleatorio |
|---|---:|---:|---:|---:|
| Sphere | 0 | 0.00020454 [0.00001740, 0.00067794] | 0.00019614 [0.00003023, 0.00192202] | 0.0117468 [0.00079528, 0.0969397] |
| Rastrigin | 0 | 0.0582081 [0.00099733, 1.20837] | 0.173241 [0.0133668, 1.07774] | 1.33435 [0.157410, 2.35374] |
| Ruta | 54 | 59 [54, 73] | 54 [54, 58] | 55 [54, 58] |

En la ruta, el óptimo exacto apareció en **2/20** corridas convencionales, **15/20** calibradas y **8/20** aleatorias. En Sphere la calibración produjo una mediana ligeramente menor y una peor corrida más alejada del óptimo. En Rastrigin la mediana favoreció al convencional. Ninguna corrida continua llegó exactamente a cero. Con estas dimensiones, semillas y parámetros puede medirse el desempeño observado, sin concluir superioridad general de un método.

Cada figura muestra la mediana del mejor costo con banda intercuartílica, la mediana de la temperatura empleada y la mediana entre semillas de las tasas de aceptación por bloques de 100 propuestas. El último bloque incluye las propuestas restantes. En el eje horizontal se incluyen las llamadas del piloto. Las curvas calibradas empiezan después de ese gasto. La tabla por semilla incluye costo, brecha, número de evaluaciones, parámetros y aceptación total.

[Sphere](sphere.png) · [Rastrigin](rastrigin.png) · [Ruta](ruta.png) · [Resultados por semilla](runs.csv)

## Reproducción

Desde la raíz de este clon en PowerShell, con el entorno local preparado:

```powershell
.\.venv\Scripts\python -m pytest -q -rs tests
.\.venv\Scripts\python -m scripts.verify_simulated_annealing --budget 2001 --seeds 20
```

Para un clon nuevo, crear el entorno con `py -m venv .venv` e instalar las dependencias con `.\.venv\Scripts\python -m pip install -r requirements.txt`. La verificación sintética usa NumPy y Matplotlib. La suite completa también importa dependencias de los módulos existentes. El comando de verificación permite elegir otra carpeta con `--output`.

Se ejecutó con Python 3.14.3, NumPy 2.4.2, Matplotlib 3.10.8, pandas 3.0.1 y pytest 9.2.0. Antes de los cambios pasaron 14 pruebas existentes. Después pasaron las mismas 14 y 27 del optimizador. En ambas ejecuciones se omitieron las dos pruebas que requieren el Excel privado y apareció un aviso de obsolescencia de la extensión externa de pytest de AnyIO.

## Siguiente etapa

La elección del evaluador industrial sigue abierta. Una alternativa estadística o física puede usarse con el mismo núcleo si se proporciona una función `objective(x)` que devuelva un costo escalar a minimizar. Las conversiones de unidades y de características que requiera cada evaluador corresponden a esa función. El generador de vecinos define qué campos pueden cambiar, y el predicado opcional `feasible(x)` permite descartar configuraciones antes de evaluarlas.

Para aplicar el algoritmo a planta, el equipo deberá acordar las variables controlables, la definición del costo y las restricciones físicas y operativas. Un adaptador podría convertir la lista de `check_load(load=x)` en un booleano. Los ensayos sintéticos verifican la búsqueda matemática. La calidad del evaluador elegido y el ahorro en horas de recocido quedan por medir con datos autorizados.

# Modelo de transferencia de calor en horno de campana: metodología, correspondencia con Yang et al. (2025) y resultados SAE 1006 vs SAE 1010

Reto Ternium, Termodinámica. Este documento explica (1) la metodología del modelo, (2) cómo cada ecuación del artículo de referencia aparece en el código y en qué se aparta, y (3) los resultados de la comparación entre los dos aceros.

Referencia: X.-J. Yang et al., *Study of heat transfer model and buried thermocouple test of bell-type annealing furnace based on thermal equilibrium*, Scientific Reports 15:13362 (2025). En adelante "el artículo"; "Ec. (n)" se refiere a su numeración.

Convención usada en todo el documento:

- **[Calculado]**: número obtenido al ejecutar el modelo (`scripts/compare_grades.py`, archivo `results/comparison/resumen.txt`).
- **[Dato Ternium]**: valor entregado por Ternium.
- **[Hipótesis]**: supuesto del modelo o valor provisional (MOCK) que debe confirmarse.
- **[Esperado]**: tendencia que se anticipa por fundamento físico, antes de verla en los cálculos.

---

## 1. Metodología

### 1.1 Qué resuelve el modelo

El modelo calcula el campo de temperatura $T(r, z, t)$ de cada rollo de una pila dentro del horno de campana. El rollo se trata como un sólido anular homogéneo equivalente: miles de capas de lámina se sustituyen por un material con conductividad radial efectiva $\lambda_r$ (que incluye las resistencias entre capas) y conductividad axial $\lambda_z = \lambda_S$ (la del acero, porque en dirección axial el calor viaja por dentro de cada lámina).

La temperatura del gas protector $T_g(t)$ es una **condición de frontera conocida**, igual que en el artículo (sección "Model establishment": la temperatura del termopar de control se toma como la del gas). El modelo no calcula la campana, el quemador ni el gas; recibe $T_g(t)$ como entrada.

### 1.2 Supuestos (los mismos cinco del artículo)

1. Simetría circunferencial: el problema es 2D en $(r, z)$.
2. Sin fuentes internas de calor.
3. Se ignora el cambio de campana durante el enfriamiento.
4. Gas a temperatura uniforme en todo el horno (alta velocidad por el ventilador).
5. Transferencia superficial por convección; por encima de 800 K se usa un coeficiente compuesto convección más radiación, Ec. (8).

Supuestos adicionales de esta implementación **[Hipótesis]**:

6. Gas protector = hidrógeno puro (Ternium no especificó la atmósfera).
7. Propiedades del acero dependientes de la temperatura (el artículo no las da).
8. La transición emulsión/óxido de la Ec. (17) usa la temperatura máxima que ha alcanzado el gas (ver 2.6).

### 1.3 Flujo de cálculo

```
Datos (furnace_parameters.py)  ─┐
Propiedades (properties.py)    ─┼─> cases.build_case()  ─> malla por rollo (mesh.py)
                                │
En cada paso de tiempo, para cada rollo:
   T_g(t) del programa (solver.gas_temperature)
   h por superficie  <- flow.py (Ec. 5-6) + convection.py (Ec. 7-8)
   λ_r por nodo      <- radial_conductivity.py (Ec. 9-17)
   ρ, c, λ_S por nodo <- properties.steel_properties(T, grado)
   Δτ estable        <- mesh.stable_time_step (Ec. 4)
   T nueva           <- solver.explicit_step (Ec. 1-3)
```

Archivos y su papel:

| Archivo | Contenido |
|---|---|
| `bell_furnace/furnace_parameters.py` | Datos del artículo (`CONTACT`, `FURNACE`, `TABLE_3`) y de Ternium (`TERNIUM`, `FURNACE_TERNIUM`, `TERNIUM_COILS`, `SCHEDULE_TERNIUM`), cada uno marcado como dato o MOCK |
| `bell_furnace/properties.py` | Tablas de hidrógeno (CoolProp) y de acero por grado (`STEEL_GRADES`) |
| `bell_furnace/mesh.py` | Malla, volúmenes y áreas de control; límite de estabilidad |
| `bell_furnace/flow.py` | Distribución del gas, Ec. (5)-(6) |
| `bell_furnace/convection.py` | Coeficiente convectivo y compuesto, Ec. (7)-(8) |
| `bell_furnace/radial_conductivity.py` | Conductividad radial equivalente, Ec. (9)-(17) |
| `bell_furnace/solver.py` | Balance térmico explícito, Ec. (1)-(4); bucle en el tiempo |
| `bell_furnace/cases.py` | Arma un caso completo (artículo o Ternium) y valida los rollos contra los límites de Ternium |
| `scripts/compare_grades.py` | Comparación SAE 1006 vs SAE 1010 (sección 3) |

### 1.4 Discretización

- Malla de nodos sobre la sección del rollo: $n_r = 21$ nodos radiales y $n_z = 31$ axiales. Los nodos de los extremos están sobre las superficies.
- Cada nodo posee un volumen de control: completo en el interior, medio volumen en bordes y un cuarto en esquinas.
- Integración explícita (Euler hacia adelante). Paso máximo de 30 s, como en el artículo, reducido automáticamente si la condición de estabilidad lo exige.
- Resultados guardados cada 300 s; el instante en que se cruza una temperatura se obtiene por interpolación lineal (`solver.time_to_reach`). Se verificó que guardar cada 60 s cambia los tiempos en menos de 0.1 min **[Calculado]**.

### 1.5 Adaptación de los datos de Ternium

| Dato Ternium | Valor original | Valor en el modelo (SI) | Uso |
|---|---|---|---|
| Diámetro interior | 508 mm | 0.508 m | Geometría del rollo |
| Diámetro exterior máximo | 1,829 mm | 1.829 m | Geometría y validación |
| Ancho | 720 a 1,320 mm | 0.72 a 1.32 m | Validación de la pila |
| Espesor de lámina | 0.30 a 3.49 mm | 0.30e-3 a 3.49e-3 m | $S$ en Ec. (9), (17) |
| Temperatura programada | 1,650 °F | 1,172 K (899 °C) | **No se usa como $T_g$** (ver nota) |
| Grados | SAE 1006, SAE 1010 | Tablas `steel_sae1006.csv`, `steel_sae1010.csv` | $\rho, c, \lambda_S$ |
| Diámetro de campana | 2,451 mm | $D$ = 2.451 m | Diámetro hidráulico, Ec. (7) |
| Altura de operación | 5,250 mm | $L$ = 5.25 m | Longitud de circulación, Ec. (7); validación de altura de la pila |
| Caudal | 52,000 ft³/min | 24.54 m³/s (caudal real) | $Q$ en Ec. (6) |
| Temperatura de saturación del gas | 1,265 °F | 958.15 K (685 °C) | $T_g$ de sostenimiento |
| Temperatura de contacto | 1,175 °F | 908.15 K (635 °C) | Punto de referencia y temperatura objetivo principal |
| Ubicación de referencia | primer rollo, 25 in desde el diámetro interior | rollo 1#, $r$ = 0.254 + 0.635 = 0.889 m | Sonda en el modelo |

Notas de interpretación **[Hipótesis]**:

- 1,650 °F se interpreta como el punto de ajuste de la campana de calentamiento (lado de los quemadores), no como la temperatura del gas que toca los rollos. Ternium indica explícitamente que el gas que no toca los rollos está a 1,265 °F, y eso es lo que el artículo usa como frontera.
- El caudal se toma como caudal real a la temperatura del gas (un ventilador mueve un volumen aproximadamente constante). En el caso del artículo se usa caudal normal (Nm³) corregido por temperatura. El código acepta ambos (`flow_basis`).
- Área de paso del gas (punto 5, no disponible): se estimó como el anillo entre la campana y el rollo más grande, $\pi/4\,(2.451^2 - 1.829^2) = 2.09$ m².
- "Primer rollo" se interpreta como el rollo inferior (1#), igual que la numeración del artículo. La altura de la sonda no se dio; se reporta a medio ancho.
- No se entregó el número de rollos, la curva temporal del horno ni los tiempos de rampa y sostenimiento. Se usa una pila de 4 rollos de 1.829 m × 1.15 m × 1.5 mm (4 rollos de 1.15 m caben en 5.25 m con las placas supuestas) y una rampa de 12 h.

### 1.6 Propiedades de los aceros

| Propiedad | SAE 1006 | SAE 1010 | Fuente |
|---|---|---|---|
| $c(T)$ | igual para ambos | igual para ambos | EN 1993-1-2 (Eurocódigo 3) |
| $\lambda_S$ a 300 K | 68.0 W/(m K) | 63.9 W/(m K) | Incropera, Tabla A.1 |
| $\lambda_S$ a 900 K | 36.2 W/(m K) | 35.3 W/(m K) | Incropera, Tabla A.1 |
| $\rho$ | 7,850 kg/m³ | 7,832 kg/m³ | Incropera, Tabla A.1 |

- SAE 1010 usa la fila AISI 1010 de Incropera.
- SAE 1006 no tiene fila publicada; se interpoló en contenido de carbono entre hierro Armco (0.015 % C) y AISI 1010 (0.10 % C), suponiendo 0.06 % C **[Hipótesis]**. El script `scripts/generate_steel_tables.py` regenera las tablas si se cambia ese valor.
- El calor específico de aceros con 0.06 a 0.10 % C lo fija la matriz de hierro, por eso se usa la misma curva **[Hipótesis]**. Esta decisión pesa mucho en los resultados (sección 3.5).
- Dureza, rugosidad y pendiente superficial por grado: Ternium no las dio. El código permite asignarlas por grado (`STEEL_GRADES[grado]['contact']`), pero hoy ambos grados usan los valores del artículo.

### 1.7 Definiciones usadas en los resultados

- **Punto frío de la pila**: mínimo de temperatura sobre todos los nodos de los cuatro rollos.
- **Tiempo de calentamiento a $T_{obj}$**: primer instante en que el punto frío de la pila alcanza $T_{obj}$.
- **Condiciones de la comparación**: misma geometría, programa de gas, caudal, parámetros de contacto y malla; lo único que cambia es el grado de **todos** los rollos. La rampa es de 12 h hasta 685 °C, seguida de un sostenimiento largo (60 h) para que el punto frío alcance todos los objetivos. El enfriamiento no forma parte de este estudio.

---

## 2. Correspondencia entre las ecuaciones del artículo y el código

### 2.1 Balance térmico de cada nodo, Ec. (1), (2), (3)

**Artículo.** Para un nodo interior (Ec. 1):

$$\rho c\,\frac{T^{i+1}_{m,n}-T^{i}_{m,n}}{\Delta\tau} = \lambda_r\Delta z\frac{T^i_{m+1,n}-T^i_{m,n}}{\Delta r} + \lambda_r\Delta z\frac{T^i_{m-1,n}-T^i_{m,n}}{\Delta r} + \lambda_z\Delta r\frac{T^i_{m,n+1}-T^i_{m,n}}{\Delta z} + \lambda_z\Delta r\frac{T^i_{m,n-1}-T^i_{m,n}}{\Delta z}$$

Las Ec. (2) y (3) agregan, en bordes y esquinas, el término convectivo $h\,A\,(T_g - T)$.

**Código.** Las tres ecuaciones son casos de una sola forma general, que es la que programa `solver.py`:

$$C_{m,n}\,\frac{T^{i+1}_{m,n}-T^{i}_{m,n}}{\Delta\tau} = \sum_{\text{vecinos}} G\,(T^i_{\text{vecino}} - T^i_{m,n}) + \sum_{\text{superficies}} h\,A\,(T_g - T^i_{m,n})$$

con

| Símbolo | Significado | Dónde se calcula |
|---|---|---|
| $C = \rho c\,V$ | capacidad térmica del nodo | `explicit_step`: `rho_c * mesh['volume']` |
| $G_r = \lambda_r A_r / \Delta r$ | conductancia radial entre nodos vecinos | `face_conductances` |
| $G_z = \lambda_z A_z / \Delta z$ | conductancia axial | `face_conductances` |
| $h A$ | coeficiente por área expuesta (cero si el nodo no está en esa superficie) | `node_heat_rates` |

Despejando $T^{i+1}$ queda `T_new = T + dt * Q / (rho_c * volume)` en `explicit_step`. Al sustituir $V = \Delta r\Delta z$, $A_r = \Delta z$ y $A_z = \Delta r$ (geometría plana por unidad de profundidad), la forma general es exactamente la Ec. (1). En un nodo de borde, $\sum hA$ contiene un término y se recupera la Ec. (2); en una esquina contiene dos ($h_E\Delta z + h_N\Delta r$) y se recupera la Ec. (3).

Diferencias respecto al artículo:

1. **Geometría axisimétrica.** Por defecto `mesh.build_mesh(geometry='axisymmetric')` calcula volúmenes y áreas por radián: $V = \tfrac12(r_e^2 - r_w^2)\,\Delta z$, $A_r = r_{cara}\,\Delta z$. Esto toma en cuenta que un anillo exterior tiene más masa y más área que uno interior. El artículo usa la forma plana (supone "la superficie cilíndrica equivalente a un plano", sección de $\lambda_r$). Con `geometry='planar'` el código reproduce la forma del artículo.
2. **Volúmenes de borde.** Los nodos de borde tienen medio volumen y las esquinas un cuarto. En las Ec. (2)-(3) del artículo el lado izquierdo no muestra ese factor; la forma con volumen de control conserva la energía de manera exacta y es la estándar del método de balance térmico.
3. **Conductividad en las caras.** Entre dos nodos con $\lambda$ distinta se usa la media armónica $2\lambda_a\lambda_b/(\lambda_a+\lambda_b)$, que equivale a dos resistencias en serie. El artículo usa un solo $\lambda$ por nodo; si $\lambda$ es uniforme ambas coinciden.

### 2.2 Estabilidad, Ec. (4)

**Artículo.** El coeficiente de $T^i_{m,n}$ en cada ecuación debe ser positivo, por ejemplo $1 - \frac{2\lambda_r\Delta z\Delta\tau}{\rho c\Delta r} - \frac{2\lambda_z\Delta r\Delta\tau}{\rho c\Delta z} > 0$. Se toma el mínimo sobre todos los nodos; para su caso resulta 30 s.

**Código.** En la forma general el coeficiente es $1 - \frac{\Delta\tau}{C}\left(\sum G + \sum hA\right)$. Que sea positivo equivale a

$$\Delta\tau < \frac{C}{\sum G + \sum hA}$$

`mesh.stable_time_step` calcula ese cociente para cada nodo (`capacity / g_sum`), toma el mínimo y lo multiplica por un factor de seguridad de 0.9. `explicit_step` usa `min(30 s, paso estable)`. Es la misma condición que la Ec. (4), aplicada nodo por nodo y recalculada en cada paso porque $\lambda_r$, $\rho c$ y $h$ cambian con la temperatura.

### 2.3 Distribución del gas, Ec. (5) y (6)

**Artículo.**

$$fd_3 = \frac{1}{2 + coNum},\qquad fd_2 = \zeta\,fd_3,\qquad fd_1 = \zeta\,fd_2,\qquad 1/\zeta = 0.909$$

$$Q_{cp1} = Q\,fd_1,\quad Q_{cp2} = Q(1-fd_1)fd_2,\quad Q_{cp3} = Q(1-fd_1)(1-fd_2)fd_3,\quad Q_{top} = Q(1-fd_1)(1-fd_2)(1-fd_3)$$

**Código.**

- `flow.flow_distribution_coefficients` generaliza la Ec. (5) a cualquier número de rollos: $fd_j = \zeta^{\,n-1-j}/(2+n)$ para $j = 1, \dots, n-1$. Con $n = 4$ da $fd_3 = 1/6$, $fd_2 = \zeta/6$, $fd_1 = \zeta^2/6$, igual que el artículo.
- `flow.channel_flows` aplica la Ec. (6) con un bucle: en cada placa sale la fracción $fd_j$ de lo que queda y el resto sube. Lo que llega a la parte superior es $Q_{top}$.
- `flow.surface_flows` asigna el caudal a cada superficie: la cara inferior del rollo $i$ recibe el flujo de la placa de abajo (cero en el rollo 1#, que se apoya en la base), la superior el de la placa de arriba (o $Q_{top}$ en el último rollo) y las caras interior y exterior el flujo que pasa a ese nivel.

**Verificación directa [Calculado]**: con $n = 4$ y $1/\zeta = 0.909$, `surface_flows` devuelve 20.2 %, 14.6 % y 10.9 % en las placas y 100 %, 79.8 %, 65.2 % y 54.3 % en las caras laterales, idénticos a la Tabla 1 del artículo.

Observación: igual que la Tabla 1, el caudal lateral se asigna tanto a la cara interior como a la exterior (ambas con "100 %" en el rollo 1#). Es la lectura literal de la tabla.

### 2.4 Coeficiente convectivo, Ec. (7) y (8)

**Artículo.**

$$h = 0.023\,\psi\,\frac{u_g^{0.8}}{D^{0.2}}\,\lambda_g\left(\frac{\mu_g}{\rho_g}\right)^{-0.8}\left[1 + \left(\frac{D}{L}\right)^{0.7}\right],\qquad k = \begin{cases} h & T_g < 800\text{ K} \\ \xi h & T_g \ge 800\text{ K}\end{cases}$$

**Código.** `convection.convection_coefficient` es la Ec. (7) término por término:

```python
0.023 * psi * u_g**0.8 / D**0.2 * k_g * (mu_g / rho_g) ** -0.8 * (1.0 + (D / L) ** 0.7)
```

y `convection.composite_coefficient` es la Ec. (8). En `solver.make_h_function`, para cada superficie de cada rollo y en cada paso:

1. caudal real $Q$ (constante con el dato de Ternium, o normal corregido por $T_g/273.15$ en el caso del artículo);
2. caudal de la superficie = $Q \times$ fracción de la Ec. (6);
3. $u_g$ = caudal / área de paso;
4. $\mu_g, \rho_g, \lambda_g$ del hidrógeno a $T_g$ (`hydrogen_properties`), lo que el artículo pide explícitamente porque estas propiedades cambian con la temperatura;
5. Ec. (7) y luego Ec. (8).

Diferencia: el artículo no publica $\psi$ ni $\xi$ (los ajustó con datos de producción). Aquí $\psi = 1.5$ y $\xi = 1.2$ son **[Hipótesis]**. Valores resultantes a 685 °C en el caso Ternium **[Calculado]**: $h$ lateral de 53.4, 44.6, 37.9 y 32.7 W/(m² K) en los rollos 1# a 4#, y de 9 a 15 W/(m² K) en las placas.

### 2.5 Conductividad radial equivalente, Ec. (9) a (17)

**Artículo.** Resistencias por unidad de área:

$$R_{S/2} = \frac{S}{2\lambda_S}\ (9),\quad R_O = \frac{b_O}{\lambda_O}\ (10),\quad R_R = \frac{2-\varepsilon}{4(1-\varphi)\varepsilon\sigma T_m^3}\ (11),\quad R_G = \frac{b}{(1-\varphi)\lambda_g}\ (12),\quad R_D = \frac{\sigma_P}{1.13\lambda_S\tan\theta}\varphi^{-0.94}\ (13)$$

$$\varphi = \frac{P}{H+P}\ (14),\qquad b = 42.7\times10^{-6}\,e^{-0.05P}\ (15)$$

$$R_\Sigma = 2R_{S/2} + R_O + \left(\frac{1}{R_R}+\frac{1}{R_G}+\frac{1}{R_D}\right)^{-1}\ (16),\qquad \lambda_r = \frac{S+b+b_O}{R_\Sigma}\ \text{o}\ \eta\frac{S+b}{R_\Sigma'}\ (17)$$

**Código** (`radial_conductivity.effective_radial_conductivity`). Se programa con conductancias ($1/R$) en lugar de resistencias, que es la misma cosa:

| Ec. | Artículo | Código |
|---|---|---|
| (9) | $2R_{S/2} = S/\lambda_S$ | `R_steel = s / k_s` |
| (10) | $R_O = b_O/\lambda_O$ | `R_oxide = oxide_thickness / k_oxide` |
| (11) | $1/R_R$ | `h_rad = 4 * (1 - phi) * epsilon * SIGMA * T_m**3 / (2 - epsilon)` |
| (12) | $1/R_G$ | `h_gas = (1 - phi) * k_g / b` |
| (13) | $1/R_D$ | `h_contact = 1.13 * k_s * tan_theta / roughness * phi**0.94` |
| (14) | $\varphi$ | `phi = pressure / (hardness + pressure)` |
| (15) | $b$ | `b = 42.7e-6 * np.exp(-0.05 * pressure)` |
| (16) | paralelo de R, G, D | `R_interface = 1 / (h_rad + h_gas + h_contact)` |
| (16) $T_g \ge 800$ K | $R_\Sigma$ | `R_high = R_steel + R_oxide + R_interface` |
| (17) $T_g \ge 800$ K | $\lambda_r$ | `k_high = (s + b + oxide_thickness) / R_high` |
| (17) $T_g < 800$ K | $\lambda_r$ | `k_low = eta * (s + b) / R_low` con `R_low = R_steel + R_interface` |

Detalles:

- En la Ec. (17) impresa, el término de radiación aparece con $T_m^4$, mientras que la Ec. (11) usa $T_m^3$. El código usa $T_m^3$, que es lo dimensionalmente correcto para un coeficiente de radiación linealizado (W/(m² K)); se interpreta como errata del artículo.
- En la rama de baja temperatura el artículo cambia $R_O$ por $R_E$ (emulsión) en la Ec. (16), pero en la Ec. (17) elimina ese término y multiplica por $\eta$. El código sigue la Ec. (17).
- `solver.make_lambda_r_function` llama a esta función para cada nodo con $\lambda_S$ del grado del rollo y $\lambda_g$ del hidrógeno evaluada a la temperatura del nodo (el gas atrapado entre capas está a la temperatura del acero).
- Ingresan como **[Dato del artículo]**: $P$ = 8 MPa, $H$ = 1,133.86 MPa, $\sigma_P$ = 3.2215 μm, $\varepsilon$ = 0.08554, $b_O$ = 10 μm. Son **[Hipótesis]**: $\lambda_O$ = 2 W/(m K), $\tan\theta$ = 0.10, $\eta$ = 1.

### 2.6 Cambio emulsión/óxido a 800 K

El artículo considera la emulsión "solo en el calentamiento por debajo de 800 K" y la capa de óxido por encima de 800 K y durante el enfriamiento. Si se usara solo $T_g$ actual, al enfriar por debajo de 800 K el modelo volvería a la emulsión, que ya se evaporó. Por eso `simulate_coil` lleva `T_g_peak` (máximo de $T_g$ hasta ese momento) y se lo pasa a la Ec. (17): una vez superados 800 K, el régimen de óxido queda fijo. Es la lectura literal del texto del artículo.

### 2.7 Qué del artículo se reproduce y qué no

- **Reproducido exactamente**: la estructura de las Ec. (1)-(17) y la Tabla 1 de distribución de flujo.
- **Reproducido cualitativamente [Calculado]**: con los datos de la Tabla 3 (`--case paper`), el rollo con peor transferencia es el 3# y el mejor el 2# al final del calentamiento (punto frío 646.9 K = 373.8 °C en 3# y 661.2 K = 388.1 °C en 2#, con un programa de gas supuesto), igual que la conclusión 2 del artículo. En el caso Ternium el rollo más lento también es el 3#.
- **No reproducido**: los valores medidos del artículo (por ejemplo, punto frío de 1# de 667.2 °C a 18.5 h). El artículo no publica su curva de $T_g(t)$, ni $\psi$, $\xi$, $\eta$, $\tan\theta$, $\lambda_O$ ni la geometría de los canales; sin esos datos la comparación numérica no es posible. Es la principal validación pendiente.

---

## 3. Resultados: SAE 1006 vs SAE 1010

Todos los valores de esta sección son **[Calculado]** salvo que se indique otra cosa. Para reproducirlos: `python scripts/compare_grades.py`.

### 3.1 Gráfica principal: temperatura contra tiempo

![Curvas de temperatura](results/comparison/R1_curvas_temperatura.png)

- Mismos ejes, unidades y condiciones de frontera para ambos aceros. Línea discontinua gris: gas (condición de frontera). Líneas gruesas: punto frío de la pila. Líneas punteadas: punto caliente.
- Líneas horizontales: temperaturas objetivo (500 a 675 °C). Marcadores: instante en que cada acero alcanza cada objetivo (círculo SAE 1006, cuadro SAE 1010).
- El recuadro amplía la zona de 635 °C (referencia de Ternium) para que se vea la separación entre las curvas.
- Panel inferior: diferencia de temperatura del punto frío, $T_{1006} - T_{1010}$. Máximo de 1.61 K a las 12.5 h, justo al final de la rampa; después se reduce hacia cero, porque ambos aceros tienden a la misma temperatura del gas.

![Tiempos contra temperatura objetivo](results/comparison/R2_tiempos_vs_objetivo.png)

- Izquierda: tiempo de calentamiento contra temperatura objetivo. Ambas curvas crecen sin límite al acercarse a la temperatura del gas (685 °C, asíntota vertical).
- Derecha: $\Delta t = t_{1010} - t_{1006}$ en minutos. Siempre positivo (SAE 1006 llega primero) y crece con la temperatura objetivo, de 4.6 min a 500 °C a 7.3 min a 675 °C.

### 3.2 Tabla de resultados

**Punto frío de la pila** (rollo 3#, el más lento en ambos aceros). Rampa de 12 h a 685 °C, sostenimiento de 60 h.

| $T_{obj}$ (°C) | $T_{obj}$ (K) | $t_{\text{SAE 1006}}$ (h) | $t_{\text{SAE 1010}}$ (h) | $\Delta t$ (min) | $\Delta t$ (%) |
|---:|---:|---:|---:|---:|---:|
| 500 | 773.15 | 22.342 | 22.420 | 4.6 | 0.35 |
| 550 | 823.15 | 26.851 | 26.938 | 5.2 | 0.32 |
| 600 | 873.15 | 33.903 | 33.999 | 5.7 | 0.28 |
| 625 | 898.15 | 39.433 | 39.533 | 6.1 | 0.26 |
| **635** | **908.15** | **42.387** | **42.490** | **6.2** | **0.24** |
| 650 | 923.15 | 48.286 | 48.394 | 6.5 | 0.22 |
| 665 | 938.15 | 57.841 | 57.955 | 6.9 | 0.20 |
| 675 | 948.15 | 70.081 | 70.202 | 7.3 | 0.17 |

635 °C = 1,175 °F, temperatura de contacto reportada por Ternium.

**Punto de referencia de Ternium** (rollo 1#, $r$ = 0.889 m, medio ancho):

| $T_{obj}$ (°C) | $t_{\text{SAE 1006}}$ (h) | $t_{\text{SAE 1010}}$ (h) | $\Delta t$ (min) |
|---:|---:|---:|---:|
| 500 | 15.409 | 15.391 | −1.1 |
| 550 | 18.661 | 18.651 | −0.6 |
| 600 | 23.935 | 23.944 | 0.5 |
| 625 | 28.220 | 28.243 | 1.4 |
| 635 | 30.546 | 30.577 | 1.9 |
| 650 | 35.235 | 35.279 | 2.7 |
| 665 | 42.903 | 42.967 | 3.8 |
| 675 | 52.849 | 52.933 | 5.1 |

Archivos: `results/comparison/tabla_tiempos.csv` y `tabla_referencia.csv`.

### 3.3 Mapas de color

**Sección $r$-$z$ del rollo más lento (3#)**:

![Mapas r-z](results/comparison/R3_mapas_rz.png)

- Filas 1 y 2: campo de temperatura de cada acero a 3, 8, 12, 20 y 30 h, con la **misma escala de color** (30 a 660 °C).
- Última columna: instante en que el punto frío de la pila llega a 635 °C (42.42 h para SAE 1006 y 42.50 h para SAE 1010, el instante guardado más cercano). La cruz marca el punto frío. En ese momento el rollo está casi uniforme, por eso toda la sección se ve del mismo color.
- Fila 3: diferencia $T_{1006} - T_{1010}$ en los mismos instantes, con escala divergente propia (±1.64 K). Es donde la diferencia entre aceros se hace visible.

**Temperatura contra posición radial y tiempo** (rollo 3#, plano medio):

![Mapa r-t](results/comparison/R4_mapa_r_t.png)

Misma escala para ambos aceros. Isoterma blanca: 635 °C. Línea cian: instante en que el punto frío de la pila llega a 635 °C. La isoterma tiene forma de arco: los bordes ($r$ = 0.254 m y $r$ = 0.915 m) llegan primero y el centro radial ($r \approx$ 0.5 m) al final, lo que ubica el punto frío en el interior del rollo.

### 3.4 Tendencias observadas

1. **SAE 1006 calienta más rápido que SAE 1010 en el punto frío, pero muy poco.** $\Delta t$ va de 4.6 a 7.3 min sobre tiempos de 22 a 70 h, es decir 0.17 % a 0.35 %. A 635 °C la diferencia es de 6.2 min en 42.4 h.
2. **$\Delta t$ absoluto crece con la temperatura objetivo y $\Delta t$ relativo disminuye.** Cerca de la temperatura del gas ambos tiempos crecen sin límite y la diferencia absoluta crece con ellos.
3. **La diferencia de temperatura es máxima al final de la rampa (1.61 K a 12.5 h)** y luego tiende a cero.
4. **Los mapas de diferencia muestran un patrón:** SAE 1006 está más caliente en el interior (rojo) y ligeramente más frío cerca de las superficies (azul) durante la rampa. El acero de mayor conductividad lleva el calor de la superficie hacia el interior con más facilidad.
5. **En el punto de referencia de Ternium (cerca de la superficie exterior) el signo se invierte a temperaturas bajas:** SAE 1010 llega a 500 y 550 °C hasta 1 min antes. Es la misma redistribución del punto 4 vista desde un punto cercano a la superficie.
6. **El rollo más lento es el 3# en ambos aceros**, igual que en el artículo: recibe el menor caudal por las placas (10.9 % y 14.6 %) y un flujo lateral de solo 65.2 %.

### 3.5 Consistencia con el fundamento físico

**[Esperado]** Para un sólido calentado desde su superficie, el tiempo para llegar a una temperatura depende de la difusividad $\alpha = \lambda/(\rho c)$ y, cuando la resistencia superficial importa, del número de Biot. SAE 1006 tiene menos carbono, mayor conductividad (+2.5 % a 900 K, +6 % a 300 K) y prácticamente la misma $\rho c$, así que debe calentar algo más rápido.

**[Calculado]** Para saber cuánto, se midió la elasticidad del tiempo de calentamiento a 635 °C (cambio porcentual del tiempo por 1 % de cambio en cada parámetro, con SAE 1006):

| Parámetro (+5 %) | Cambio en $t_{635}$ | Elasticidad |
|---|---:|---:|
| $\rho$ (equivale a $\rho c$) | +4.17 % | +0.83 |
| $h$ (todas las superficies) | −3.09 % | −0.62 |
| $\lambda_r$ | −0.81 % | −0.16 |
| $\lambda_z$ | −0.65 % | −0.13 |

Interpretación:

- El tiempo de calentamiento lo controlan la **capacidad térmica** ($\rho c$) y la **convección del gas** ($h$). Las conductividades del acero pesan cuatro a cinco veces menos.
- Diferencias entre grados: $\lambda_r$ +1.8 a +4 % y $\lambda_z$ +2.5 a +6 % a favor de SAE 1006; $\rho$ +0.23 % (SAE 1006 un poco más pesado, lo que lo hace más lento); $c$ igual.
- Estimación lineal: $\Delta t/t \approx -0.16(1.8\%) - 0.13(2.5\%) + 0.83(0.23\%) \approx -0.4\%$ a alta temperatura. Lo calculado es −0.24 %: mismo signo y mismo orden de magnitud. El resultado es consistente con la física del modelo.
- $\lambda_r$ cambia menos que $\lambda_S$ porque solo parte de la resistencia radial es el acero: las resistencias de la interfaz entre capas (gas, radiación) no dependen del grado. Solo la conductancia por contacto, Ec. (13), escala con $\lambda_S$.

**Límites de comportamiento de los tiempos de calentamiento:**

- **Límite superior de tiempo.** Durante el sostenimiento el punto frío se acerca al gas de forma casi exponencial, $T_g - T \propto e^{-t/\tau}$, con $\tau$ = 17.35 h (SAE 1006) y 17.36 h (SAE 1010). El tiempo crece como $t \approx t_0 + \tau\ln\!\big(\Delta T_0/(T_g - T_{obj})\big)$ y tiende a infinito cuando $T_{obj} \to T_g$ = 685 °C. Por eso cada 10 °C adicionales cerca de 685 °C cuestan mucho más que a 500 °C (de 665 a 675 °C se pasa de 57.8 a 70.1 h).
- **Límite de la diferencia entre aceros.** Como $\tau$ casi no cambia entre grados, $\Delta t$ crece lentamente (logarítmicamente) con el objetivo y nunca supera unos minutos en el rango útil. Aun si la conductividad fuera el único parámetro relevante, la elasticidad combinada ($-0.29$) aplicada a la diferencia de conductividades (2 a 6 %) da una diferencia de 0.6 a 1.7 %, es decir menos de 1 h en 42 h.
- **Lo que cambiaría la conclusión.** El parámetro con más peso, $\rho c$, es el que hoy es igual para ambos grados. Si Ternium tuviera datos de $c(T)$ propios por grado, o si los grados tuvieran dureza o rugosidad distintas (Ec. 13-14), la diferencia podría ser mayor que la calculada.

### 3.6 Conclusiones

1. **[Calculado]** Con los datos actuales, la diferencia de tiempo de calentamiento entre SAE 1006 y SAE 1010 es menor a 0.4 % (4.6 a 7.3 min en 22 a 70 h). SAE 1006 llega antes al objetivo en el punto frío.
2. **[Calculado]** Para un objetivo de 635 °C (1,175 °F) en el punto frío de la pila: 42.39 h (SAE 1006) y 42.49 h (SAE 1010), con rampa de 12 h y gas a 685 °C.
3. **[Calculado + fundamento]** La diferencia es pequeña porque el calentamiento lo controlan la capacidad térmica y la convección del gas, que son prácticamente iguales para ambos aceros. La conductividad del acero, que es lo que distingue a los grados, tiene poco peso.
4. **[Hipótesis]** Para fines de programación del horno, ambos grados pueden tratarse con el mismo ciclo; la decisión depende mucho más del caudal, del rollo más desfavorable (3#) y de la temperatura objetivo que del grado.
5. **[Hipótesis]** La incertidumbre de los parámetros MOCK ($\psi$, $\xi$, $\tan\theta$, área de paso, tiempos de rampa) es mucho mayor que 0.4 %. La diferencia entre grados es real dentro del modelo, pero menor que la incertidumbre del modelo mismo. Los valores absolutos de tiempo deben tomarse como orden de magnitud hasta validar con mediciones.

### 3.7 Datos que mejorarían los resultados

| Dato | Efecto |
|---|---|
| Curva $T_g(t)$ o tiempos reales de rampa y sostenimiento | Tiempos absolutos de calentamiento |
| Mediciones de termopar en un ciclo real | Ajustar $\psi$ y $\xi$ como en el artículo |
| $c(T)$ por grado | El parámetro de mayor peso (elasticidad 0.83) |
| Dureza, rugosidad o $\tan\theta$ por grado | Distinguir los grados en la Ec. (13)-(14) |
| Área de paso del gas, número de rollos por carga | Valores reales de $h$ |

---

## 4. Cómo reproducir

```bash
python scripts/generate_steel_tables.py              # tablas de los aceros
python scripts/compare_grades.py                     # sección 3 (results/comparison)
python scripts/compare_grades.py --targets 600 635 660 --hold-hours 80
python scripts/run_mock_case.py                      # caso Ternium completo
python scripts/run_mock_case.py --grade SAE1010 --animate
python scripts/run_mock_case.py --case paper         # caso del artículo
```

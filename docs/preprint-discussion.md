# Resultados y discusión metodológica — CX-Net

> Borrador del informe. Fuente primaria de datos y decisiones: `docs/lab-notebook.md`
> (2026-09-16 a 2026-10-12). Varias afirmaciones de versiones anteriores están retractadas
> (tabla de la sección 9). Enfoque: informe metodológico y negativo. **Estado: exploratorio.**
> Las enumeraciones de 64 patrones (sección 6 y sección 11, tarea corregida) usan 3 semillas por
> patrón; el resto de análisis, una o pocas corridas. Solo algunos criterios se fijaron antes de
> ver los resultados (se indica en cada caso). La revisión crítica interna está en
> `docs/expert-review.md`.

## 1. Resumen

Preguntamos si una red recurrente cuya única restricción estructural es la
topología sináptica real del núcleo del sistema de dirección de cabeza de
*Drosophila* (152 neuronas, 9.160 aristas; male-cns:v1.0; `EPG`, `EPGt`,
`PEN_a`, `PEN_b`, `PEG`, `Delta7`), con la magnitud de cada peso fijada al
número real de sinapsis y el signo como único parámetro de arista, recupera por
descenso de gradiente el neurotransmisor real de cada neurona (H1).

**No encontramos evidencia de que lo haga, y el diseño no permitía encontrarla.**
(En todo el texto, «integra» significa pérdida held-out < 0.466 —la de recordar la fase
inicial sin integrar— y pendiente > 0.5 entre el cambio decodificado y el cambio real de rumbo.)
El valor del trabajo está en cinco hallazgos metodológicos, cada uno de los
cuales invalida una versión anterior del experimento:

1. La tarea original de integración de rumbo era resoluble de forma trivial: un
   decodificador constante superaba a todos los modelos entrenados (0.18 frente
   a 0.58-0.72), y la referencia de "azar" usada inicialmente era una fase
   aleatoria, no una solución trivial (sección 4).
2. La geometría angular del anillo, medida con coordenadas de sinapsis, confirma en
   MaleCNS un espejo entre hemisferios ya descrito en la anatomía y que ninguno de
   nuestros mapeos previos recogía (sección 5).
3. Con una tarea bien planteada, el modelo original no puede integrar el rumbo
   ni siquiera con los signos reales en ningún régimen dinámico explorado (588
   evaluaciones); con ganancias por tipo celular sí, pero **la asignación real
   de neurotransmisor no es especial**: 28 de los 64 patrones de signo por tipo
   integran por mayoría de 3 semillas y la real queda en el puesto 14 de 64 por
   held-out medio (p = 0.22; el rango según el criterio de agregación es 14-28,
   sección 6; con una sola semilla, 26/64 y puesto 22), apoyándose en
   tasas con signo (desviaciones respecto a un nivel basal nulo) que hacen el
   signo del peso poco identificable (sección 6). Repetida la enumeración sobre la tarea OU
   corregida (3ª revisión; sección 11), la conclusión se mantiene: 37 de 64 patrones integran por
   mayoría de 3 semillas (32 en las 3) y la real queda en el puesto 30 de 64 (p = 0.47).
4. Un aprendiz de signos genérico no encuentra esa solución (sección 7).
5. El neurotransmisor es función exacta del tipo celular en este subcircuito
   (Delta7 = glutamato, resto = acetilcolina), lo que reduce la "química real"
   a un dato de tipo y invalida los cálculos de potencia por neurona (sección 3).

## 2. Diseño

- **Modelo.** Red recurrente de tasas con `tanh`, magnitudes de peso fijas
  (conteo de sinapsis normalizado por neurona destino), signo por arista
  parametrizado por `tanh(sign_param)` y leído como `sign(sign_param)`.
- **Tarea original.** Integración de velocidad angular inyectada en PEN; el
  rumbo se decodifica como vector poblacional de EPG/EPGt sobre `ring_angle`;
  pérdida `1 − cos`. Variantes: `hold_prob` (tramos de quietud), `perturb_amp`
  (ruido de entrada), `sign_reg` (regularización de polarización del signo).
- **Tarea anclada** (sección 4): fase inicial aleatoria θ₀ inyectada como pulso
  en EPG/EPGt; objetivo θ₀ + ∫velocidad; `max_av=0.15`, `hold_prob=0.3`.
- **Evaluación de H1.** Acuerdo entre `sign(sign_param)` y el signo del
  neurotransmisor de la neurona de origen; test de permutación de etiquetas
  entre neuronas (2000 permutaciones, una cola).
- **Neurotransmisor anotado.** `predictedNt` de male-cns:v1.0 es una predicción a partir
  de EM (Eckstein et al., 2024), no una medida directa; para estos tipos es consistente
  con lo descrito en la literatura (por verificar con referencias concretas). Se asume
  además que el signo lo fija la neurona presináptica (ley de Dale) y que acetilcolina
  excita y glutamato inhibe (vía GluClα en este circuito); no se modelan co-transmisión
  ni dependencia del receptor. Hecho estructural: **el neurotransmisor anotado es
  función exacta del tipo** (tabla siguiente).

| tipo | neuronas | neurotransmisor |
|---|---|---|
| Delta7 | 42 | glutamato (100%) |
| EPG, EPGt, PEG, PEN_a, PEN_b | 46+4+18+20+22 = 110 | acetilcolina (100%) |

**Consecuencia para H1.** Como el neurotransmisor es función del tipo, y que Delta7 es
glutamatérgico e inhibidor en este circuito es un hecho descrito desde antes, «recuperar
el neurotransmisor» equivale en la práctica a recuperar un solo bit: *Delta7 inhibe, el
resto excita*. Aun confirmada, H1 aportaría poco sobre la biología; la pregunta con
contenido es si la función y la topología restringen el signo *más allá* de lo ya
conocido. Este trabajo la aborda, sin respuesta concluyente (secciones 6-8).

## 3. H1 sobre la tarea original: nulo, y poco informativo

**Polarización.** Un test de acuerdo solo es informativo si la red se
compromete con un signo por arista. Los modelos iniciales tenían polarización
débil (`mean_abs_sign` 0.31-0.36). Cuatro vías intentaron corregirlo, con
protocolo multi-semilla (3-8 semillas, held-out fijo): hiperparámetros de
dinámica (sin efecto), `hold_prob` (0.349-0.364, insuficiente), `hold_prob` +
`perturb_amp` (0.465 ± 0.163, bimodal; 1 de 8 semillas polariza) y `hold_prob` +
`sign_reg=0.05` (0.687 ± 0.123; 7 de 8 semillas cruzan el umbral de referencia
`frac_polarized_gt_0.9 > 10%`). Solo `sign_reg` lo resuelve de forma limpia; se
añadió tras observar el techo de polarización, no formaba parte del diseño
original.

**H1 con `sign_reg`.** El acuerdo con el neurotransmisor real es
prácticamente el del azar:

| semilla | acuerdo | media nula | z | p (una cola) |
|---|---|---|---|---|
| 0 | 0.4965 | 0.4978 | −0.28 | 0.618 |
| 1 | 0.5193 | 0.5071 | +2.49 | 0.0055 |
| 2 | 0.5029 | 0.4999 | +0.57 | 0.294 |
| 3 | 0.5038 | 0.4987 | +1.05 | 0.152 |
| 4 | 0.4957 | 0.4987 | −0.44 | 0.686 |
| 5 | 0.4758 | 0.4938 | −3.16 | 0.999 |
| 6 | 0.4855 | 0.4929 | −1.33 | 0.913 |
| 7 | 0.4932 | 0.4963 | −0.65 | 0.742 |

Agregado: Stouffer z = −0.62 (p = 0.73), Fisher p = 0.26. La semilla 1
(p = 0.0055; probabilidad de un mínimo tan bajo entre 8 p-valores bajo la nula
≈4.3%; por debajo del umbral de Bonferroni 0.00625) tiene un efecto diminuto
(+1.2 pp), no se replica y la semilla 5 muestra la desviación opuesta de mayor
magnitud. Un resultado aislado análogo con `perturb_amp` (p = 0.0005) se
retractó por criterio de polarización, no por comparaciones múltiples. Los 8
modelos comparten las mismas etiquetas: no son 8 pruebas independientes de H1.

**Por qué este nulo es poco informativo.** (i) La tarea era resoluble de forma
trivial y los modelos ni siquiera la igualaban (sección 4); no hay razón para
que sus signos reflejen una restricción funcional. (ii) **El control positivo de
potencia por neurona (v1) no es aplicable.** El test está calibrado (falso
positivo 0.049-0.052 con etiquetas señuelo), pero la potencia se midió sembrando
señal en fracciones de neuronas independientes (152 unidades) o de aristas; como
el neurotransmisor es función del tipo, una señal real estaría estructurada en
6 unidades, una de ellas con 42 neuronas, y las cifras de potencia (p. ej. "80%
para +3.3 pp") no son transferibles. (iii) El test de permutación baraja
etiquetas entre neuronas, rompiendo la estructura por tipo. Los acuerdos
observados (0.476-0.519) no muestran estructura por tipo y no hubo falsos
positivos, pero una nula formalmente correcta sería sobre patrones de signo por
tipo (sección 6), y con solo 6 tipos ese test tiene resolución mínima: hay
2⁶ = 64 patrones posibles, así que el p más pequeño por coincidencia exacta es
1/64 = 0.016, y si solo se contrasta «Delta7 inhibe» frente a los seis lugares
posibles de una única etiqueta distinta, 1/6 = 0.17.

## 4. Trampas de la tarea de integración

| referencia (held-out, `hold_prob=0.3`) | pérdida absoluta |
|---|---|
| Modelos `sign_reg` entrenados | 0.58-0.72 |
| Fase aleatoria (la "azar" de v1) | ≈1.0 |
| **Decodificador constante en 0** | **0.179** |
| Integrador perfecto con fase inicial arbitraria | 0.999 |

1. **Referencia de azar inadecuada.** La afirmación de v1 "la red aprende
   mejor que el azar (0.6 frente a 0.85)" comparaba con una fase aleatoria.
   Un decodificador constante da 0.18: los modelos son 3-4 veces peores. Con
   una pérdida invariante al desfase constante, los entrenados dan 0.08-0.36
   frente a 0.06 del decodificador constante.
2. **Tarea sin ancla.** La tarea pedía el rumbo absoluto (parte de 0) sin
   ninguna pista de dónde está el 0; ni un integrador perfecto con fase inicial
   arbitraria baja de ≈1.0. Con signos reales fijos, la pérdida original fue
   1.006 frente a 0.983 ± 0.103 de 500 asignaciones barajadas (p = 0.58); con
   la pérdida invariante al desfase y tres mapeos de `ring_angle`, 0.459 / 0.139
   / 0.115 frente a medias barajadas 0.155 / 0.130 / 0.125 (p = 0.98 / 0.77 /
   0.55), todos por encima del suelo trivial (0.060).
3. **Tarea anclada.** Se sortea θ₀ por ensayo y se inyecta como pulso de 20
   pasos en EPG/EPGt. Referencias en el held-out: recordar θ₀ sin integrar
   **0.466**; decodificador constante 1.105. Criterio de "integra": pérdida
   < 0.466 y pendiente (cambio decodificado / cambio real) > 0.5.
4. **Piloto sin parámetros por tipo.** Dos pilotos (`ring_sign` ±1): 0.452 y
   0.477, pendiente ≈ 0, error de anclaje 7-10°. Anclan y mantienen la fase,
   no integran. Una sonda con velocidad constante mostró activación saturada
   (r ≈ 1.0) y un desplazamiento del bump a un punto fijo en lugar de una
   velocidad proporcional; subir la ganancia de entrada de 10 a 30 no cambia
   nada.

## 5. Geometría del anillo medida en sinapsis

MaleCNS no expone la fase (cuña) de cada neurona en el EB (solo zonas radiales
`EBr*`), pero neuPrint da coordenadas 3D de sinapsis. Con 261.546 sinapsis en
el EB de 110 neuronas (`extract_eb_angles.py`), un ajuste de plano por PCA
(89% de la varianza en 2 componentes) y la media circular ponderada por
confianza, cada neurona queda en una cuña (concentración 0.89-0.99).

- Los EPG de L recorren el anillo en un sentido (L1…L8 ≈ 137°, 177°, 216°,
  265°, 313°, 356°, 47°, 90°) y los de R en el **sentido contrario** (R1…R8 ≈
  115°, 65°, 22°, 335°, 291°, 239°, 194°, 156°): un espejo, con pasos de ≈45°.
  Las 16 posiciones (L y R juntos) se intercalan alrededor del anillo,
  alternando hemisferio, con un paso medio de ≈22.5° entre posiciones
  consecutivas (no un desfase constante de 22° entre glomérulos homólogos:
  la diferencia Ln−Rn crece con n, ≈22° + 90°·(n−1), y solo coincide con 22°
  para n=1 y n=5). EPGt (glomérulo 9) queda en 113-148°, la fase del glomérulo 1.
- Es consistente con lo descrito para el sistema (cada hemisferio muestrea el anillo
  completo a ≈45° con desfase L/R de 22.5°, Hulse et al., 2021, y la proyección alterna
  de las cuñas del EB a los dos hemisferios del PB viene de trabajos anteriores) y **no
  es un hallazgo anatómico nuevo**: la aportación es una medición reproducible en
  MaleCNS y la corrección de nuestros mapeos previos, que colocaban L y R en mitades
  del círculo (el original) o suponían el mismo sentido en ambos hemisferios (el
  «intercalado» descartado el 2026-09-16).
- **Validación con los PEN, separando regiones (2026-09-28, respuesta a la revisión
  externa; `src/cx_net/pen_offset_by_region.py`, `results/pen_offset_by_region.json`).**
  La primera validación (no usados para construir el mapa) mezclaba, sin darse cuenta,
  sinapsis de todas las regiones al pesar los vecinos EPG de cada PEN, y daba un
  desfase diluido de ≈ −6° / +5.5° entre hemisferios. `fetch_adjacencies` da un
  desglose por ROI que esa versión descartaba. **Corrección (3ª revisión, 2026-09-29,
  `results/region_analysis.json`):** este párrafo decía que «casi todas las sinapsis
  EPG→PEN caen en PB»; es falso. De las ≈12.3k sinapsis EPG→PEN, solo 1.439 (12 %) están en
  el PB y ≈10.9k en el EB; las PEN→EPG sí están casi todas en el EB (≈24k). Por eso el
  cálculo sin filtrar da ≈6°: la mayor parte del peso EPG→PEN cae en el EB, donde la fase
  de entrada coincide con la de la propia PEN. Repetido SIN filtrar por región
  (comprobación pedida por el revisor) da |desfase| medio 6.3° (PEN_a) y 6.2° (PEN_b),
  0 % de neuronas en 40-60°, es decir, reproduce el valor antiguo: la dilución es real y no
  un artefacto de la validación anterior. Separando la fase de entrada (vecinas EPG presinápticas, solo
  PB) de la fase de salida (vecinas EPG postsinápticas, solo EB) por neurona PEN, el
  desfase sale mucho mayor y consistente entre las dos poblaciones: PEN_a −54.1°(L) /
  +54.3°(R) (n=10+10), PEN_b −51.4°(L) / +50.4°(R) (n=11+11) — cerca de un glomérulo
  (≈45°), no de los ≈6° del grafo mezclado, y con la misma antisimetría L/R perfecta.
  La fase propia de cada PEN (medida de sus propias sinapsis en el EB, independiente
  de la conectividad) coincide con la fase de salida hasta ~1°. Confirma la sospecha
  de la revisión: al sumar todas las regiones en `build_topology_only_graph`, el peso
  EPG→PEN del EB (sin desfase) domina sobre el del PB y diluye el desfase de fase; medido
  por región, el circuito sí desplaza la fase en una magnitud anatómicamente razonable.
  Esto es una medida de fase por vecindad, no un resultado de dinámica.
- El sentido de giro global es arbitrario en el método (`ring_sign`); ambos
  sentidos dan resultados equivalentes en la comprobación de realizabilidad.
- Limitaciones del método: el ajuste de plano por PCA supone un EB aproximadamente
  plano y un muestreo angular parejo de sinapsis; no se contrastó con el recuento de 16
  cuñas ni con la misma medición en hemibrain frente a las tablas publicadas.

## 6. Realizabilidad y (falta de) especificidad de la química real

**Régimen dinámico.** Con signos reales fijos y la tarea anclada, se exploraron
tres rondas de hiperparámetros sin entrenar (588 evaluaciones: `recurrent_gain`
0.5-16, `tau` 2-20, ganancia de entrada 1-300, ganancia de pista 3-10,
activación `tanh` y rectificada, ambos sentidos de giro; conjunto de ajuste
distinto del held-out). La mejor pérdida fue 0.470-0.493 (referencia del
conjunto de ajuste 0.537) con pendiente ≈ 0; las configuraciones con pendiente
≈ 1 tenían recurrencia débil y perdían el ancla (pérdida 0.76-1.02). Mantener
el bump exige recurrencia fuerte; moverlo con velocidad proporcional, régimen
casi lineal. **En este modelo la red con signos reales no integra.**

**Repetido sobre el grafo por compartimento (2026-09-28, tras confirmar que el desfase de
los PEN estaba diluido, sección 5).** Con el mismo protocolo (rondas 1 y 2, 288 + 100 = 388
evaluaciones sin entrenar) pero sobre `data/raw/malecns_pb_eb` (solo sinapsis PB+EB, sin
NO/FB/LAL/CentralBrain-unspecified), el resultado es el mismo: mejor pérdida 0.491-0.517,
pendiente siempre entre −0.08 y +0.03 en las 10 mejores configuraciones de cada ronda
(`results/regime_search_pbeb.csv`, `results/regime_search_r2_pbeb.csv`,
`src/cx_net/regime_search.py` con `graph_data_dir`). **Un mecanismo de rotación
anatómicamente más fuerte (desfase de PEN ≈50° en vez de ≈6°) no es suficiente por sí
solo: la red con signos reales y magnitudes fijas, sin ganancias por tipo entrenables,
sigue sin integrar en ningún régimen dinámico explorado.** Esto acota la hipótesis de la
sección 11: no basta con "el grafo diluye el mecanismo"; hace falta además normalizar o
reponderar las magnitudes (conteo de sinapsis) para que ese mecanismo más fuerte se
traduzca en dinámica, algo que las ganancias por tipo entrenadas sí hacen.

**Con ganancias por tipo entrenables, sobre el mismo grafo por compartimento (2026-09-28,
`data/interim/launch_pbeb.sh`; mismos hiperparámetros que la fila `tanh, ring_sign=+1` de
la tabla de arriba, sin re-ajustar para este grafo).** Resultado inesperado: la asignación
real va PEOR, no mejor, que con el grafo mezclado: held-out 0.281 / 0.299 / 0.339 (media
0.306, frente a 0.106-0.124 antes) y pendiente 0.42 / 0.40 / 0.36 (media 0.39, frente a
0.75-0.84 antes). **Con el criterio de "integra" (held-out < 0.466 y pendiente > 0.5),
ninguna de las 3 semillas integra sobre este grafo** (el held-out mejora respecto a la
memoria, pero la pendiente se queda corta). Los 4 barajados por neurona sobre el mismo
grafo tampoco integran (held-out 0.43-0.60, pendiente entre −0.02 y +0.07). **Sí hay
especificidad** (corrección de la 3ª revisión): la real se separa de los barajados
(held-out 0.31 frente a 0.43-0.60; pendiente 0.39 frente a ≤ 0.07); solo se queda por debajo
del umbral arbitrario de «integra» (pendiente > 0.5). Lectura: quitar el 13.4% del peso sináptico que cae fuera
de PB/EB (en NO, FB, LAL(L/R) y CentralBrain-unspecified) no "purifica" el mecanismo de
rotación, lo debilita: esas conexiones descartadas por parecer ajenas al núcleo PB-EB
aparentemente contribuyen a sostener la dinámica que las ganancias por tipo explotan sobre
el grafo completo. **Contradice la hipótesis de partida** ("el grafo mezclado diluye y por
tanto debilita el mecanismo; sin diluir, debería integrar mejor o necesitar menos ayuda de
las ganancias"); el resultado es el opuesto. Salvedad importante: los hiperparámetros
dinámicos (`recurrent_gain`, `tau`, `in_gain`, `cue_gain`) se eligieron para el grafo
mezclado (sección 6, "criterio A") y no se volvieron a ajustar para este grafo — no se
puede descartar que un régimen distinto sí integre; hace falta repetir la búsqueda de
régimen tunable (no solo sin entrenar) o una nueva sección de la búsqueda de hiperparámetros
sobre este grafo antes de concluir que el circuito PB+EB por sí solo es insuficiente.

**Qué falta en PB+EB: las sinapsis PEN↔PEN del NO (3ª revisión, 2026-09-29; `src/cx_net/region_analysis.py`,
`results/region_analysis.json`).** El NO solo contiene sinapsis entre PEN (PEN_a→PEN_a, PEN_b→PEN_b,
PEN_a↔PEN_b: 15.440 sinapsis, ≈11 % del peso total; el resto de regiones fuera de PB/EB suman ≈2 %).
*Ablación sin reentrenar* sobre los 3 modelos reales del grafo completo, con la normalización
original: quitar el NO hace que ninguno integre (held-out 0.43, pendiente 0.10, 0/3); quitar FB,
LAL(L), LAL(R) o CentralBrain-unspecified por separado no cambia nada (0.11-0.12, 3/3). *Reentrenando*
la asignación real (3 semillas, misma receta, `data/interim/launch_regions.sh`): grafo completo SIN NO:
held-out 0.331 / 0.296 / 0.357, pendiente 0.42 / 0.43 / 0.33 (0/3 integran, como PB+EB); PB+EB+NO:
held-out 0.127 / 0.104 / 0.113, pendiente 0.77 / 0.82 / 0.80 (3/3 integran, como el grafo completo).
Es decir, las sinapsis PEN↔PEN del NO son necesarias y, junto con PB+EB, suficientes para la solución
entrenada con la asignación real. Es una comprobación de necesidad/suficiencia de una región, no una
explicación mecanística de por qué la recurrencia PEN↔PEN ayuda. Esto sustituye a la lectura anterior
(«NO/FB/LAL aportan algo más específico»): FB, LAL y CentralBrain no intervienen.

**Búsqueda de régimen sin entrenar, reajustada para PB+EB (2026-09-28). Retirada la etiqueta «salvedad
cerrada» (3ª revisión): esta búsqueda no descarta nada sobre la versión ENTRENADA.** Rejilla propia (`recurrent_gain` hasta 24, 1.5× el máximo probado con el grafo
mezclado; `tau` ∈ {5,10,20}; `in_gain` ∈ {10,30,100,300}; 216 evaluaciones,
`results/regime_search_pbeb_r3.csv`; 388 + 216 = 604 en total): mismo negativo en todo el rango — mejor pérdida 0.498,
pendiente entre −0.14 y +0.01 en las 20 mejores configuraciones. Ningún régimen explorado (604 evaluaciones sin
entrenar en total sobre este grafo) logra que los signos reales solos integren sobre PB+EB.
**Pero ese mismo negativo ya se daba en el grafo completo, donde la versión entrenada sí
integra, así que no dice nada sobre por qué falla la versión entrenada; y las ganancias por
par de tipos pueden multiplicar hasta ×148 (log acotado a ±5), así que `recurrent_gain`
difícilmente era la explicación.** Queda
abierta una pregunta más interesante que la de partida: las ganancias por tipo entrenadas SÍ
resuelven la tarea sobre el grafo mezclado pero fallan sobre PB+EB, así que el papel de
NO/FB/LAL no parece ser solo aportar más recurrencia (ya descartado aquí) sino algo más
específico de esas conexiones que ni las ganancias por tipo compensan sobre el grafo
restringido — sin explicación mecanística todavía.

**Parámetros por tipo.** Se añadieron ≈43 parámetros compartidos por tipo
celular, nunca por arista: una ganancia positiva por par de tipos
origen→destino, un sesgo por tipo y una escala global de la entrada. Con signos
reales fijos (solo se entrenan esos parámetros; control de realizabilidad, sus
valores no se reutilizan):

| variante | held-out | pendiente |
|---|---|---|
| `tanh`, `ring_sign=+1`, 3 semillas de entrenamiento | 0.108 / 0.124 / 0.106 | 0.82 / 0.75 / 0.84 |
| `tanh`, `ring_sign=−1` | 0.125 | 0.87 |
| rectificada, `ring_sign=−1` / `+1` | 0.172 / 0.467 | 0.73 / 0.00 |

**Barajados por neurona e intercambio parcial** (`ring_sign=+1`, mismo
protocolo): 12 asignaciones barajadas dan 0.444-0.598 (0/12 integran, pendiente
máx. 0.04) y el intercambio de etiquetas en 10 / 25 / 50% de las neuronas (≈4 /
≈10 / ≈22% de aristas cambiadas) da 0.463-0.507 (0/9 integran). El estadístico
preespecificado da p = 0.077, el mínimo alcanzable con 12 barajados. Las 3
corridas reales comparten una sola asignación (difieren solo en la semilla de
entrenamiento): demuestran reproducibilidad, no 3 asignaciones distintas.

**Enumeración de patrones de signo por tipo.** Como el neurotransmisor es
función del tipo, la hipótesis natural es un signo por tipo: 2⁶ = 64 patrones,
de los que la asignación real (solo Delta7 inhibe) es uno. Se entrenaron los 64
con el mismo protocolo, primero con una sola corrida por patrón (semilla 0) y
después replicados con 2 semillas más (2026-09-22, tras la revisión externa:
"¿cuántas semillas por patrón antes de creer un ranking?"), 3 en total:

- **Con una sola semilla, 26/64 integran y la real es el puesto 22 (p =
  22/64 = 0.344).** Con las 3 semillas, el número de patrones que integran
  depende del criterio de agregación: 20/64 lo hacen en las 3 semillas,
  28/64 por mayoría (≥2 de 3) y 36/64 en al menos una. El puesto de la real
  por held-out medio es 14/64 (p = 0.22); por mediana, 19/64 (p = 0.30); por
  la mejor semilla, 28/64 (p = 0.44) — se da el rango, no un único valor. La
  real integra en las 3 semillas (held-out 0.106-0.124, media 0.113). De los
  otros 63 patrones, **16 no son unánimes entre semillas** (confirma que una
  sola corrida por patrón es ruidosa, como ya se sospechaba); de esos, en 4
  el voto de mayoría difiere del de la semilla 0: 3 pasan de "no integra" a
  "integra" (máscaras 14, 15, 57) y 1 al revés (máscara 45). Patrones
  biológicamente absurdos la siguen superando (p. ej. "solo EPG inhibe":
  held-out 0.063/0.061/0.589, media 0.238, dominada por una semilla que no
  converge; "EPG+EPGt inhiben": media 0.065).
- Fracción de patrones que integran (mayoría de 3 semillas) según el signo del
  tipo (excitador / inhibidor, de 32 cada uno): Delta7 8 / 20; EPG 10 / 18;
  EPGt 15 / 13; PEG 12 / 16; PEN_a 15 / 13; PEN_b 22 / 6. Descriptivo, sin
  test. Delta7 inhibidor ayuda pero no es necesario ni suficiente; PEN_b
  excitador es el sesgo individual más fuerte.
- **Interacción Delta7 × EPG × PEN_b (propuesta por la revisión externa, verificada
  aquí sobre `types_analysis_3seeds.json`).** Con PEN_b excitador (32 patrones), la
  fracción que integra según el signo de Delta7 y EPG es: ninguno inhibe 0/8; solo
  Delta7 inhibe 8/8; solo EPG inhibe 8/8; ambos inhiben 6/8. Con PEN_b inhibidor solo
  integran 6/32 en total, y de ellos casi todos tienen Delta7 inhibidor (2/8 y 4/8 en
  las mismas dos celdas). Es decir, con PEN_b excitador la regla empírica es «al
  menos uno de EPG o Delta7 inhibe»: 22 de 24 patrones integran, frente a 0 de 8 cuando
  ninguno inhibe. (Corrección del revisor: no es «retroalimentación negativa del bucle»;
  cuando inhiben los dos, el bucle EPG→Delta7→EPG es positivo —negativo por negativo— y
  aun así integran 6 de 8.) Es una regularidad descriptiva, sin explicación mecanística,
  que reformula el resultado negativo como una afirmación comprobable.
- Los 12 barajados fallan porque rompen la coherencia por tipo, no porque la
  química real sea especial: **la lectura de "suficiencia funcional" de la v1 se
  retracta.**

**Mecanismo probable.** Con `tanh` las tasas van en (−1, 1) y el nivel basal es 0: se
interpretan como desviaciones respecto a una actividad basal. En la red que integra con
la asignación real (semilla 0), Delta7 está por debajo de la basal el 89% del tiempo (fracción > 0 =
0.11; media −0.17): una neurona inhibidora que baja su actividad desinhibe a sus dianas,
con efecto neto excitador. Es una lectura legítima si la neurona real tuviera una
actividad basal tónica suficiente, pero el modelo ni la representa ni la limita, y hace
que el signo del peso deje de ser identificable a partir de la función: muchos patrones
de signo se vuelven funcionalmente equivalentes (comprobación descriptiva sobre 5
ensayos).

**Enunciado formal de la simetría, y por qué NO se aplica tal cual a este modelo
(propuesto por la revisión externa; formalizado y verificado el 2026-09-28,
`src/cx_net/sign_symmetry_analysis.py`).** Con una activación impar (`tanh(-x) =
-tanh(x)`) y sesgo entrenable, la dinámica `r_{t+1} = tanh(W r_t + I_t + b)` es
invariante ante `r → Dr`, `W → DWD`, `b → Db`, `I → DI` para cualquier `D` diagonal con
entradas ±1 ("switching" de grafos con signo) **si las ganancias que absorben `D` pueden
tomar cualquier signo**. Pero en este modelo las ganancias son `exp(log_pair_gain)`,
siempre positivas, una por cada PAR DE TIPOS (36 parámetros), y el signo va por tipo (no
por arista). Para que `D` (uniforme dentro de cada tipo, `D_tipo ∈ {±1}⁶`) siga siendo una
simetría de ESTA parametrización, hace falta `D_tipo(a) · D_tipo(b) = +1` para todo par de
tipos `(a,b)` con al menos una arista — si no, la ganancia compensatoria tendría que ser
negativa, y el modelo lo prohíbe. La matriz de conectividad entre los 6 tipos es un grafo
COMPLETO (las 36 combinaciones dirigidas, incluidos los 6 lazos consigo mismo, tienen peso
> 0; verificado también sobre el grafo PB+EB de la sección 5, igual de completo), así que
esa condición para todos los pares fuerza `D_tipo` a ser constante: **los únicos `D`
compatibles con las ganancias positivas son el trivial y el global (los 6 tipos a la vez);
no existe ninguna simetría EXACTA de signo por tipo no trivial en este modelo.** (Comprobación
por fuerza bruta sobre los 2⁶=64 vectores `D` posibles: solo esos 2 son compatibles, en
ambos grafos.)

**Esto corrige la lectura anterior de este párrafo (versión previa a 2026-09-28), que
enunciaba la simetría sin comprobar la restricción de positividad.** Consecuencia: los 64
patrones NO son gauge-equivalentes EXACTOS entre sí. **Matiz de la 3ª revisión (2026-09-29):
de «no hay simetría exacta» no se sigue «las diferencias son dinámicas reales».** El script
solo comprueba peso > 0 por par de tipos; con muy pocas sinapsis un par cuenta igual (los
pares PEN→Delta7 suman 12-14 aristas y 14-15 sinapsis), y las matrices 6×6 muestran que PEG
y EPGt tienen acoplamiento de salida efectivo ≤ 0.1 hacia casi todos los tipos
(`results/sign_symmetry_pairs.json`; ninguna semilla real lleva más de 1 par a la cota
inferior de la ganancia, pero varias ganancias de PEN→Delta7 saturan en la cota superior,
×148, sobre pesos casi nulos). Existen, por tanto, simetrías aproximadas. Empíricamente,
cambiar solo el signo de EPGt y PEG mueve el held-out una mediana de 0.055 (frente a una sd
global de 0.17), aunque en 3 de los 16 grupos cambia el veredicto, con rangos de hasta 0.56.
Además 16 de 63 patrones no son unánimes entre semillas: parte de la variación es ruido de
optimización. Las diferencias de rendimiento (held-out de 0.05 a 0.77) están, pues,
dominadas por el signo de Delta7, EPG, PEN_a y PEN_b, pero no se puede afirmar que todas sean
«dinámicas reales». La tabla Delta7 × EPG × PEN_b
de arriba sigue siendo una regularidad empírica genuina —no una consecuencia de la
simetría formal— sobre qué patrones sostienen la dinámica, y queda sin explicación
mecanística. El argumento de simetría sí seguiría aplicando en un modelo con ganancias
libres por arista y de signo libre (no el de este proyecto); con ganancias por tipo y
restringidas a ser positivas, que es lo que se entrenó aquí, no aplica. Con el patrón sin
ninguna inhibición (que no integra), Delta7 tiene fracción > 0 = 1.00. Por tanto la
solución con la asignación real no debe leerse como una solución mecanísticamente
biológica.

**Validación mecanística (descriptiva; `bump_check.py`).** Con velocidad constante
dentro del rango de entrenamiento (±0.01 a ±0.08 por paso durante 90 pasos), la red con
la asignación real desplaza el bump en el sentido correcto, de forma antisimétrica, pero
**no proporcional: la ganancia por velocidad es sigmoidal**, no una recta por el origen.
Por paso: ×1.65 a 0.01, ×1.24 a 0.02, ×0.80 a 0.04, ×0.46 a 0.08 — sobrepasa el
desplazamiento esperado a velocidades bajas y satura a las altas (esperado ±7.2 rad,
decodificado −2.8 / +3.3). La recta ajustada por el origen (pendiente 0.54, R² = 0.84,
la cifra citada en secciones previas de este documento) resume mal esa forma: hay que
leerla como una cifra agregada de una respuesta no lineal, no como una ganancia
constante. Otros patrones que integran (pendiente agregada 0.75-0.76, R² 0.92-0.93)
muestran el mismo patrón de sobrepaso/saturación en su tabla de ganancias por velocidad
(`results/bump_check.json`). La red sin inhibición no responde (desplazamiento ≈ −0.2
rad a todas las velocidades). La localización del bump (|Σ r e^{iθ}| / Σ|r| sobre
EPG/EPGt) es 0.62 en la solución con la asignación real y 0.34-0.37 en «solo EPG
inhibe» y «EPG+EPGt inhiben». Con una velocidad constante grande (±0.3 durante 180
pasos) ningún modelo desplaza el bump. La pendiente de las secciones anteriores mide la
integración agregada en el rango de entrenamiento, no un integrador general ni lineal.
Se comprobó que no es un artefacto del decodificador.

**Perfil del bump, kymographs y memoria de fase (`bump_profile.py`,
`results/bump_profile.json`, figura 6).** Respuesta a la objeción de que una pérdida baja
no prueba la existencia de un bump localizado. Con velocidad 0, alineando la actividad de
las 50 neuronas de compás (EPG/EPGt) con la fase decodificada (12 contenedores de 30°, ≈4
neuronas por contenedor, media poblacional restada), el ancho a media altura es de 167°
(asignación real), 122° («solo EPG inhibe»), 135° («EPG+EPGt inhiben») y 133° (sin
inhibición); una coseno tendría 180°. Es decir, el «bump» es ancho y de perfil suave, no
un paquete estrecho, y la resolución (30°) es gruesa. La fracción de neuronas saturadas
(|r| > 0.95) es 0.00 en los tres modelos que integran y 0.83 en el que no integra: este
último es un patrón binario congelado (media anillo a +1 y la otra media a −1), no un
bump graduado. En los kymographs (figura 6) los tres modelos que integran arrastran la
banda de actividad y la fase decodificada en el sentido correcto, con la respuesta
sigmoidal (sobrepaso a baja velocidad, saturación a alta) descrita arriba, algo menos
marcada en los otros dos patrones (pendiente agregada 0.75-0.76);
el que no integra no se mueve con ninguna velocidad. La memoria de una fase arbitraria
(8 fases de pista, velocidad 0) no discrimina: los cuatro modelos, incluido el que no
integra, sostienen la fase con error circular medio de 0.06-0.15 rad a t = 60 y 0.18-0.40
rad a t = 199. Lo que separa a los integradores del resto es la respuesta a la
velocidad y la actividad graduada, no la retención de fase. Salvedades: una sola corrida
por modelo, resolución angular gruesa (50 neuronas), y no se comparó con el perfil de
un modelo de anillo construido a mano.

**Magnitudes libres por arista, signos fijos (`edge_gain`, `analyze_edge_gain.py`,
`results/edge_gain*_analysis.json`).** Prueba propuesta por la revisión externa: ¿son las
magnitudes rígidas (conteo de sinapsis) lo que impide integrar? Se añadió una ganancia
positiva por arista (9.160 parámetros) sobre la receta por tipo, con los signos fijos. Con
la misma tasa de aprendizaje que el resto (ronda 1) la asignación real deja de integrar
(held-out 0.470-0.475, pendiente ≈ 0; con ganancias por tipo era 0.106-0.124) y de 6
barajados solo uno integra (0.236): es un fallo de optimización, no un resultado sobre la
biología, porque la solución por tipo está contenida en este espacio (ganancia 1). Con una
tasa 10 veces menor para las ganancias por arista (ronda 2; elegida después de ver el
fallo, en el sentido que favorece a la real) la real integra en 3 de 3 semillas (0.078-0.103,
pendiente 0.80-0.91) y ninguno de 6 barajados ni el patrón sin inhibición lo hace
(0.35-0.58). Pero las ganancias por arista apenas se mueven (percentiles 5-95 entre 0.8 y
1.3) y no se poda ninguna arista, así que esto es casi el resultado por tipo con una mejora
pequeña (0.08-0.10 frente a 0.11-0.12), no una prueba de que las magnitudes rígidas sean el
problema: con libertad grande no se optimiza, y con libertad pequeña no cambia nada. La
hipótesis «fuerza funcional ≠ conteo de sinapsis» sigue sin poder separarse de la de
«identidad del signo» con este diseño; la comparación a nivel de tipo (ganancias por tipo:
la real integra, los barajados no) es la evidencia disponible.

**Tasas no negativas (sigmoide).** Con `activation="sigmoid"` (tasa basal 0.5), mismos
hiperparámetros y protocolo (600 épocas), los signos reales dan held-out 0.303
(`ring_sign=−1`, pendiente 0.33) y 0.470 (`+1`, pendiente 0.01), y 4 barajados
0.536-0.955: los reales quedan por delante de los barajados, pero ninguno cumple el
criterio de «integra». Con esos hiperparámetros la comprobación de realizabilidad no se
cumple, así que no se enumeraron los 64 patrones; haría falta una búsqueda de régimen
propia. El resultado es no concluyente.

## 7. Aprendibilidad

Con la tarea anclada original (no se ha repetido con la tarea OU corregida de la sección 11), con signos por arista y parámetros por tipo entrenados a la vez desde un inicio
neutro (ganancias 1, sesgos 0, signos ≈ N(0, 0.1); sin usar nada de los pasos
anteriores), cuatro pilotos de una semilla (`ring_sign` ±1 × `sign_reg` 0.05 /
0; 1000 épocas) dan held-out 0.458-0.561 y pendiente −0.10 a 0.00: convergen al
óptimo local "memoria sin integrar". Existe una solución en el espacio de
búsqueda (0.11 con signos reales) que el descenso de gradiente no encuentra.
Hipótesis sin verificar: plateau ancho entre las soluciones de memoria y de
integración (BPTT de 200 pasos con dinámica casi saturada). No se probó
currículo, otro inicio, otra tasa de aprendizaje ni más épocas. Al no aprender el
aprendiz ninguna solución que integre, H1 (que converja a la química real) no
es evaluable.

**Repetición sobre la tarea OU corregida (2026-10-12, preregistrada; `results/sign_learner_v3.json`).** Con la receta que sí
converge (800 épocas, pérdida incremental), un aprendiz de 6 signos por tipo (los mismos 6 parámetros que la enumeración) integra
en 7/10 semillas con inicio neutro (3/10 si se binarizan los signos a ±1) y en 6/10 con inicio binario aleatorio, pero
**recupera la asignación real en 0/20** (patrón «solo Delta7 inhibe»; azar de referencia 1/37). Aprende 9 patrones distintos en 10
corridas por brazo, con Delta7 inhibidor en 6/10 y 3/10. Con un signo por arista (6 semillas) el acuerdo con el signo real es
0.51 (azar). Arrancando en la real, las 6 corridas siguen integrando pero ninguna conserva ese patrón (5 derivan a «Delta7 + EPGt +
PEG inhiben», 1 a «Delta7 + PEG»; Delta7 sigue inhibidor): la real no es un atractor. Lectura: con esta tarea el gradiente no
prefiere la química real, de acuerdo con que 37 de 64 patrones integran; no es una prueba de que ningún aprendiz pueda encontrarla.
La etiqueta formal del criterio preregistrado depende de si «integra» exige también la versión binarizada (caso «no identificable»
con el modelo continuo; caso «fallo de optimización» para A si se exigen ambos; B queda sin etiqueta); lo sólido es el 0/20.

## 8. Interpretación

- **H1 tal como se formuló no está ni confirmada ni refutada**: el aprendiz no
  llega a una solución funcional, y la solución funcional existente no depende
  de la química real. El nulo de la sección 3 no informa sobre la biología.
- **Un bit conocido.** Con el neurotransmisor función del tipo, el contraste de fondo es
  «Delta7 inhibe» (conocido) frente a las otras 63 combinaciones. Que la tarea no lo
  seleccione, con este modelo, dice que la tarea y el modelo no imponen esa
  restricción, no que la biología no la imponga. Con la tarea OU corregida (sección 11) el
  cuadro se matiza: Delta7 inhibidor es casi suficiente (30 de 32 patrones integran) y bastante
  necesario (7 de 32 con Delta7 excitador), así que la tarea sí selecciona ese bit conocido;
  pero, fijado ese bit, los otros cinco apenas importan (30 de 32 integran) y la real es el
  puesto 28 de esos 32. La tarea recupera lo que ya sabíamos y nada más específico.
- **Sesgo de selección a favor de la real.** Los hiperparámetros dinámicos se eligieron
  en zonas donde la asignación real funcionaba; eso favorecería a la real frente al resto
  de patrones. Que aun así quede en el puesto 14 (con una sola semilla, 22) refuerza el
  nulo en vez de debilitarlo.
- **Lo que sí se sostiene:** (a) las cinco trampas metodológicas; (b) un modelo
  de tasas con signo puede resolver la tarea con patrones de signo por tipo
  muy distintos, así que **la tarea de integración de rumbo con activación
  `tanh` no selecciona la química**; (c) la fragilidad frente a perturbaciones
  parciales de etiquetas por neurona (≈4% de aristas ya la destruye) puede
  reflejar afinado fino o mera sensibilidad del optimizador de 43 parámetros;
  no se distinguen.
- **Consecuencia práctica para trabajos similares:** antes de interpretar un
  acuerdo o un desacuerdo de signo, comprobar (1) que existe una solución trivial
  que el modelo no supera, (2) que la tarea da pista de fase, (3) que el modelo
  con los signos anotados resuelve la tarea, (4) que el resultado no depende de
  tasas con signo sin nivel basal, (5) si el neurotransmisor anotado es función de
  una variable de tipo, y (6) que la red que «integra» responde de forma
  proporcional a la velocidad y no solo en el decodificador.

### Posicionamiento y trabajo previo (referencias por verificar antes de citar)

No se ha hecho una revisión sistemática. Trabajos de partida que un lector esperaría ver
discutidos: la descripción anatómica del circuito (Wolff et al., 2015; Hulse et al.,
2021; Turner-Evans et al., 2017); los modelos de anillo atractor basados en el conectoma
del sistema de dirección de cabeza, entre ellos Kakaria y de Bivort (2017), Turner-Evans
et al. (2020) y Pisokas et al. (2020), y Green, Adachi, Shah, Hirokawa, Magani y Maimon
(2017, *Nature* 546:101, «A neural circuit architecture for angular integration in
Drosophila»), que obtienen dinámica de bump con parámetros ajustados y una inhibición de
Delta7 específica; Chang, Huang y Lo (2023, *J. Comp. Physiol. A*, «Global inhibition in
head-direction neural circuits: a systematic comparison between connectome-based spiking
neural circuit models»), que comparan explícitamente la inhibición global por Delta7
frente a la de las ring neurons — directamente relevante porque nuestro modelo excluye
las ring neurons; Noorman, Hulse, Jayaraman, Romani y Hermundstad (2024, *Nat. Neurosci.*
27(11):2207-2217, «Maintaining and updating accurate internal representations of
continuous variables with a handful of neurons»), un modelo mínimo de referencia; Vafidis,
Owald, D'Albis y Kempter (2022, *eLife* 11:e69841, «Learning accurate path integration in
ring attractor models of the head direction system»), sobre aprendizaje de la integración
en el anillo de la mosca; Beiran y Litwin-Kumar (2025, *Nat. Neurosci.* 28:2561-2574,
«Prediction of neural activity in connectome-constrained recurrent networks»), sobre la
identificabilidad de la dinámica a partir solo del conectoma, muy cercano a nuestra
pregunta central; modelos entrenados con restricción de conectoma (Lappalainen et al.,
2024) y la sensibilidad topológica de esos modelos (Dhiman, 2026); la predicción de
neurotransmisor (Eckstein et al., 2024); y el conectoma MaleCNS (Berg et al., 2026). La
comparación de nuestra dinámica con la de esos modelos ajustados a mano (¿qué parámetros
e inhibición requieren?) es la vía más directa para entender por qué el modelo original
no integra.

Estado de verificación (2026-09-21, ampliado 2026-09-28 tras la revisión externa).
Existencia de las referencias confirmada en la web del editor: Pisokas, Heinze y Webb
(2020), *eLife* 9:e53985, «The head direction circuit of two insect species»;
Turner-Evans et al. (2020), *Neuron* 108(1):145-163, «The neuroanatomical ultrastructure
and function of a biological ring attractor»; Franconville, Beron y Jayaraman (2018),
*eLife* 7:e37017, «Building a functional connectome of the Drosophila central complex»;
Turner-Evans, Wegener, Rouault, Franconville, Wolff, Seelig, Druckmann y Jayaraman
(2017), *eLife* 6:e23496, «Angular velocity integration in a fly heading circuit»;
Chang, Huang y Lo (2023), *J. Comp. Physiol. A*, «Global inhibition in head-direction
neural circuits: a systematic comparison between connectome-based spiking neural circuit
models»; Noorman, Hulse, Jayaraman, Romani y Hermundstad (2024), *Nat. Neurosci.*
27(11):2207-2217; Vafidis, Owald, D'Albis y Kempter (2022), *eLife* 11:e69841; Beiran y
Litwin-Kumar (2025), *Nat. Neurosci.* 28:2561-2574. Su contenido NO está verificado
contra el texto completo (salvo lo indicado explícitamente abajo). Lo
poco que se leyó de la página de Pisokas et al. corrige mi descripción anterior: son
neuronas de integración y disparo (LIF), no un modelo de tasas, y los pesos no se ajustan a
mano sino que se optimizan (agrupando varias optimizaciones con k-means); los pesos
absolutos se declaran arbitrarios y lo que emerge es un patrón de eficacias *relativas* entre
tipos (E-PG→P-EN/P-EG fuertes; P-EN→E-PG y P-EG→E-PG débiles) y una velocidad de giro del
bump que crece de forma exponencial con la estimulación unihemisférica de las P-EN. Esto es
coherente con que las ganancias por tipo, y no el signo, sean lo que hace falta, pero no lo
demuestra. Falta leer Kakaria y de Bivort (2017, *Front. Behav. Neurosci.* 11:8; modelo de integración y
disparo del puente protocerebral completo, recapitula la dinámica de anillo atractor), Kim et
al. (2017, *Science* 356:849; es un trabajo experimental con calcio + optogenética, no un
modelo mecanicista: muestra excitación local e inhibición global) y Prinz, Bucher y Marder
(2004, *Nat. Neurosci.* 7:1345; circuitos con parámetros muy distintos dan actividad similar),
que respalda que varios patrones de signo resuelvan la tarea. Corrección (2026-09-28, revisión
externa): en el cuaderno (2026-09-16) cité dos veces "Green et al. (2017)" para Delta7
inhibidor / resto excitador en modelos analíticos y luego, al no localizarlo por su cuenta,
un revisor interno marcó la cita como posiblemente errónea. El artículo sí existe: Green,
Adachi, Shah, Hirokawa, Magani y Maimon (2017), *A neural circuit architecture for angular
integration in Drosophila*, *Nature* 546:101 — uno de los trabajos fundacionales sobre PEN.
Se retira la advertencia anterior; la cita es correcta y debe usarse en el posicionamiento
(añadida arriba).

## 9. Retractaciones respecto a la v1

| afirmación de la v1 | estado |
|---|---|
| "La red aprende la tarea mejor que el azar (0.6 vs. 0.85)" | Retractada: referencia errónea; los modelos son 3-4× peores que un decodificador constante (sección 4). |
| "Control de potencia: el test detecta ≥ +3.3 pp con ≥ 80%" | Retractada: siembra por neurona/arista, no aplicable a una señal estructurada por tipo (sección 3). Solo se mantiene la calibración del falso positivo. |
| "Signos reales no superan a barajados: 1.006 vs 0.983" (primera lectura) | Superada: prueba injusta sin ancla; ver sección 4. |
| "El mapeo intercalado es el conforme a la anatomía" | Corregida: L y R recorren el anillo en sentidos opuestos (sección 5). |
| "La química real integra y las barajadas no (suficiencia funcional)" | Retractada: 28/64 patrones por tipo integran (3 semillas); la real, puesto 14 de 64 (sección 6). |
| "~440 evaluaciones de régimen" | Corregida: 588. |
| «Tasas negativas no fisiológicas» (versión intermedia) | Matizada: son desviaciones respecto a un nivel basal nulo; el problema es la falta de identificabilidad del signo (sección 6). |

## 10. Limitaciones

- **Enumeración de 64 patrones con 3 semillas (2026-09-22):** 16 de 63 patrones
  distintos del real no son unánimes entre semillas (de esos, en 4 el voto de
  mayoría difiere del de la semilla 0), lo que confirma que una sola corrida es
  ruidosa (limitación que se mantenía abierta). La real es estable en las 3
  (0.106-0.124, siempre integra). Un solo `ring_sign` (+1) en las corridas de
  refuerzo y de enumeración; el criterio de "integra" sigue siendo el umbral
  binario arbitrario (M5) y las 3 semillas no sustituyen a un test formal con
  más réplicas. Además, la selección del checkpoint dentro de cada corrida
  (el lote de 16 ensayos de menor pérdida, `best_state`) es en sí misma una
  fuente de varianza entre semillas: no se comparó con seleccionar por media
  móvil (EMA) o con un conjunto de validación aparte.
- **El «p» de los puestos no es un p-valor de contraste.** Es la fracción p = puesto/64 de la
  real dentro de una enumeración exhaustiva: los 64 patrones no son una muestra aleatoria de una
  nula, los hiperparámetros se eligieron con la real, y el puesto cambia con el criterio de
  agregación y con la semilla (en la tarea corregida: puesto 30 por pérdida media de 3 semillas,
  36-37 por semilla individual). Es un descriptivo del tamaño del conjunto de soluciones, no una
  prueba de significación.
- **Sensibilidad al umbral de «integra» (tarea corregida, post hoc).** El recuento depende sobre
  todo del umbral de pendiente: 37 patrones integran con pendiente > 0.5 o > 0.7, 32 con > 0.8 y
  20 con > 0.9; con > 0.8 la real (pendiente 0.74) deja de integrar en las 3 semillas. El umbral
  de pérdida (0.538, 0.3 o 0.2) no cambia el recuento. La conclusión «la real no es especial» es
  robusta a esos umbrales; «la real integra» no lo es.
- **El aprendiz de signos sobre la tarea corregida** (sección 7) se probó con 6 signos por tipo (20 corridas) y por arista
  (6); sigue sin explorarse otro `ring_sign`, otras tasas de aprendizaje, más épocas ni hiperparámetros ciegos a la real, y
  los hiperparámetros dinámicos se eligieron con la real; 10 semillas por brazo, sin cálculo de potencia. La etiqueta del
  criterio preregistrado depende de una ambigüedad declarada (continuo frente a binarizado).
- **Hiperparámetros dinámicos** (`recurrent_gain=2`, `tau=10`, `in_gain=10`,
  `cue_gain=10`) elegidos en zonas seleccionadas con ayuda de los signos reales
  (criterio A): grado de libertad del investigador, declarado. Más ≈30
  configuraciones de las fases anteriores y umbrales de polarización fijados sin
  análisis previo de potencia; ningún p-valor aislado debe leerse como
  confirmatorio.
- **Tasas con signo (`tanh`)**: la comprobación con sigmoide (tasa basal 0.5, sección 6)
  no fue concluyente con los mismos hiperparámetros; la rectificada fue inestable. No
  hay un test con tasas no negativas que cumpla la realizabilidad.
- **Nula del test de H1 a nivel de neurona** (sección 3) y potencia no medida
  para señal por tipo.
- **Un solo subcircuito y una sola tarea**: 152 neuronas del núcleo, sin ring
  neurons ni fan-shaped body; entrada de velocidad sintética inyectada en PEN;
  normalización por neurona destino que borra ganancias relativas entre tipos.
- **Muestreo pequeño en pilotos**: el paso de aprendibilidad son 4 corridas de
  una semilla; los modelos de la sección 3 usan 8 semillas pero sobre una tarea
  trivial.
- **Solución con magnitud efectiva**: forzar los signos entrenados a ±1 degrada
  su pérdida (0.58-0.72 → 0.70-0.79): "magnitud fija" se cumple solo a medias
  en la solución aprendida.
- **Neurotransmisor anotado, no medido; supuestos de signo.** `predictedNt` es una
  predicción a partir de EM; se asume ley de Dale y acetilcolina→excita,
  glutamato→inhibe. No se modelan la co-transmisión ni la dependencia del receptor
  (p. ej. receptores muscarínicos inhibidores o receptores de glutamato excitadores).
- **«Integra» es un umbral binario arbitrario** (held-out < 0.466 y pendiente > 0.5,
  fijado antes de la enumeración). El p exacto usa el rango continuo del held-out, pero
  no hay intervalos de incertidumbre por corrida y la diferencia entre puestos vecinos
  (p. ej. el 22 frente al 15) está dentro del ruido de optimización (las réplicas de la
  real varían 0.106-0.124).
- **Las fracciones por tipo (sección 6) son marginales de patrones no independientes**:
  cada patrón entra en seis marginales; no son efectos causales de cada tipo.
- **Integración no lineal**: respuesta sigmoidal a la velocidad (sobrepaso a baja
  velocidad, saturación a alta; sección 6), no una ganancia constante; el perfil del
  bump y su memoria de fase sí se caracterizaron (`bump_profile.py`, figura 6), pero
  no se compararon con redes con integradores ajustados a mano.
- **Entorno**: con torch 2.4.1 (Python del sistema) el gradiente de
  `atan2(0,0)` es `nan`; el entorno del proyecto (`.venv`, torch 2.14) da 0. Los
  entrenamientos por defecto deben ejecutarse con el `.venv`
  (reproducción exacta verificada). Los dos pilotos de la tarea anclada se
  ejecutaron con el Python del sistema (la pista evita el caso `(0,0)`).

## 11. Trabajo futuro

Prioridades, de mayor a menor para cerrar la lógica del estudio:

1. **(Hecho el 2026-10-12, sección 7.)** El aprendiz de signos sobre la tarea OU corregida no recupera la real (0/20 con
   6 signos por tipo) y la real no es un atractor (D). Queda abierto separar «la tarea no lo selecciona» de «el aprendiz no
   lo encuentra»: requeriría otra tarea o un criterio que sí discrimine químicas (prioridad 3 y 4).
2. **Más réplicas y un contraste formal.** 5-10 semillas para los patrones cercanos al umbral y
   para la real, intervalos por patrón y una nula explícita (p. ej. patrones barajados por
   arista con el mismo número de inhibidores), en vez de rangos de puestos.
3. **Hiperparámetros sin ayuda de la real.** Reajustarlos por patrón (o con un criterio ciego a
   la asignación) para descartar que el sesgo a favor de la real o en su contra explique el puesto.
4. **Variar el modelo.** Tasas no negativas con basal explícita, `ring_sign` distinto de +1,
   neuronas del anillo y cuerpo en abanico; y la comparación con modelos de anillo ajustados a mano.
5. **Verificar la bibliografía** del posicionamiento (varias referencias no se han leído en texto completo).

Lista detallada y tareas ya hechas:

- Buscar un régimen dinámico donde la asignación real integre con tasas no negativas
  (sigmoide, rectificada estable o desviaciones con nivel basal explícito) y solo
  entonces repetir la enumeración de 64 patrones; con los hiperparámetros actuales la
  sigmoide no lo logra.
- Validar el mapeo de fase aplicando el mismo procedimiento a hemibrain y comparándolo
  con las tablas publicadas; comparar la dinámica con modelos de anillo ajustados a mano.
- ~~Recalcular el grafo por compartimento (PB vs. EB) para el desfase de los PEN~~:
  hecho el 2026-09-28 (sección 5); confirma la dilución (repetido sin filtrar el 2026-09-29: ≈6°). ~~Reconstruir
  `graph_no_sign.csv` por compartimento y repetir la búsqueda de régimen sin
  entrenar~~: hecho el mismo día (sección 6); el mecanismo más fuerte no basta por sí
  solo. ~~Repetir con ganancias por tipo entrenables sobre ese grafo~~: hecho el mismo
  día (sección 6) — resultado inesperado, la real va peor (ninguna de 3 semillas
  integra) con los hiperparámetros tal cual, sin volver a ajustarlos para este grafo.
  ~~Repetir la búsqueda de régimen reajustada (recurrent_gain hasta 24) específicamente
  para PB+EB~~: hecho el mismo día (sección 6) — mismo negativo, pero no cierra la salvedad: no
  es un problema de `recurrent_gain` insuficiente (etiqueta retirada en la 3ª revisión). Resuelto el 2026-09-29: lo que falta en PB+EB son las sinapsis PEN↔PEN del NO (ablación y reentrenamiento, sección 6); FB, LAL y CentralBrain no intervienen.
- ~~Formalizar la simetría de signo (`r → Dr`, `D` diagonal ±1) y calcular las clases
  de equivalencia de los 64 patrones~~: hecho el 2026-09-28
  (`src/cx_net/sign_symmetry_analysis.py`, sección 6) — con resultado negativo pero
  informativo: dado que las ganancias del modelo son positivas y por PAR DE TIPO (no
  libres por arista) y la conectividad entre los 6 tipos es un grafo completo, no existe
  ningún `D` no trivial compatible con la parametrización; **los 64 patrones no son
  gauge-equivalentes exactos; de ello no se sigue que todas las diferencias sean reales
  (simetrías aproximadas por pares casi nulos y ruido de optimización; ver sección 6)**. Sigue pendiente
  incluir formalmente la entrada de velocidad (signo opuesto en PEN L/R) en el
  argumento, y explicar mecanísticamente (no solo describir) por qué el bucle
  EPG↔Delta7 con PEN_b excitador selecciona los patrones que integran.
- ~~Selección de checkpoint por media móvil (EMA) en vez del lote de menor pérdida
  (`best_state`)~~: implementado el 2026-09-28 (`train.py`, `checkpoint_selection="ema"`,
  compatible con el comportamiento anterior por defecto). Falta repetir con ella la
  enumeración completa de 64 patrones (ahora mismo se validó junto con el rediseño de
  tarea, ver abajo, no en aislamiento). 4ª-5ª semilla para los patrones cerca del
  umbral (mask14, mask15, mask45, mask57) sigue sin hacerse.
- ~~Rediseñar la tarea para que discrimine mejor entre integradores parciales y
  perfectos~~: implementado el 2026-09-28 (`task.py`: `velocity_process="ou"` con
  `tau_v` configurable, en vez de ruido blanco independiente por paso; `model.py`:
  `state_noise_std` para ruido de proceso; entrenamientos con `T` mayor para que el
  rumbo cubra varias vueltas con frecuencia). Con la tarea nueva (T=300, `tau_v`=30,
  `state_noise_std`=0.02), un integrador con ganancia 0.54 (el que antes pasaba el
  criterio de "integra" con pérdida 0.138) da pérdida 0.85 (frente a 0.98 de "memoria
  sin integrar" y 1.12 del decodificador constante) — mucho peor que integrar de
  verdad — y hace falta una ganancia dentro de ±5% de 1.0 para pérdida < 0.2
  (`results/task_v2_reference.json`); acota el hueco que señaló la revisión, aunque el integrador ideal (cumsum) no es una red de esta clase de modelos y no demuestra que la solución exista en el espacio de búsqueda. **Pero no se ha podido entrenar nada sobre ella en tres intentos.** Con
  3 semillas reales + 4 barajados, mismos hiperparámetros dinámicos de siempre: 600
  épocas no converge (held-out 0.83-1.06, pendiente ≈0 en las 7); 1400 épocas (más del
  doble) tampoco (held-out 0.82-1.04, pendiente ≈0); un currículo de 1400 épocas
  (primeras 800 interpolando desde T=200/`tau_v`=5, casi tan fácil como la tarea
  vieja, hasta el objetivo T=300/`tau_v`=30) tampoco (held-out 0.93-1.07, pendiente
  entre −0.04 y +0.02) — ni siquiera la real se distingue de los barajados en ninguno
  de los tres intentos, así que ni más épocas ni un programa de dificultad creciente
  son el problema por sí solos. La solución existe en el espacio de búsqueda (la
  referencia de ganancia ≈1.0 da pérdida ≈0), pero el descenso de gradiente con esta
  receta de hiperparámetros dinámicos (heredada de la tarea vieja, sin volver a
  ajustar) no la encuentra con ninguno de los tres programas de entrenamiento
  probados. **En ese momento el rediseño de tarea era una limitación abierta, no un
  resultado**: cierra el hueco teórico que motivó el cambio, pero no se pudo usar para
  repetir ningún análisis de este informe (cuaderno, 2026-09-28 (6), 2026-09-29 y
  2026-09-29 (2)). Pendiente, sin intentar: una búsqueda de régimen (sin entrenar,
  como la de la sección 6) específica para esta tarea, u otra tasa de aprendizaje /
  inicialización. (Superado el 2026-09-29: la tarea OU corregida de abajo sí converge y permitió
  repetir la enumeración completa de los 64 patrones.)
  No se implementaron los "barridos de velocidad constante" como parte de la tarea de
  entrenamiento — `bump_check.py` ya los usa como diagnóstico aparte (sección 6).
- **Tarea OU corregida (3ª revisión, 2026-09-29; `launch_taskv3.sh`, `task_v3_reference.py`,
`transfer_check_v3.py`).** Las tres tentativas anteriores fallaron probablemente por VELOCIDAD, no por
correlación temporal: la velocidad que ve la red (filtrada con τ=10) tenía media 0.078 y p90 0.19
(37 % del tiempo por encima de 0.08) frente a 0.019 / 0.047 / 1 % en la tarea antigua, y ninguna red
había movido el bump a más de ≈0.037-0.055 rad/paso (`bump_check.json`). Con `max_av=0.04`, `T=500`,
`tau_v=50` la velocidad vuelve al rango antiguo (p90 0.055; 43 % de los ensayos llegan a una vuelta
completa; mediana 0.40 vueltas netas) y se usa una pérdida sobre incrementos en ventanas de 20 y 50 pasos
más anclaje (`increment_loss`), que no exige acumular la ganancia a lo largo de varias vueltas; el ruido
de estado del held-out tiene ahora semilla propia. Control previo, sin reentrenar: las 3 redes reales
entrenadas en la tarea antigua ya dan pendiente 0.73-0.78 en esta (pérdida incremental 0.20-0.21,
frente a 0.54 de «recordar theta0»). Reentrenando (800 épocas, 3 reales + 4 barajados por neurona):
**las 3 reales convergen** (pérdida incremental 0.115 / 0.122 / 0.126; pendiente 0.74 / 0.73 / 0.75;
error de anclaje 14-16°) y **los 4 barajados no** (0.35-0.46; pendiente 0.10-0.20). La pérdida circular
absoluta sigue alta en las reales (0.45-0.54): se acumula la desviación de fase a lo largo de las vueltas,
como ya avisaba el revisor. Límites: solo 4 barajados por neurona (no se ha repetido la enumeración de
los 64 patrones por tipo sobre esta tarea), y sigue sin demostrarse que la tarea discrimine entre
patrones por tipo. Lo que queda establecido es que el fallo anterior era de velocidad y no de la receta
ni del presupuesto.
- **Enumeración de los 64 patrones sobre la tarea OU corregida (2026-10-09/12; `launch_types_v3.sh`,
`analyze_types_v3.py`, `results/types_analysis_v3.json`).** Criterio fijado antes de ver resultados:
«integra» = pérdida incremental held-out < 0.538 («recordar theta0» en esta tarea) y pendiente > 0.5,
voto de mayoría de 3 semillas (189 corridas, ninguna excluida). **37/64 patrones integran por mayoría
(32/64 en las 3 semillas); la real (pérdida incremental 0.121, pendiente 0.74, integra en 3/3) es el
puesto 30 de 64 (p = 0.47).** La conclusión de la sección 6 se mantiene en la tarea corregida: la real
integra pero no es especial (29 patrones la superan, con pendiente ≈0.95 frente a 0.74). Por tipo (de 32
patrones, integran con el tipo excitador / inhibidor): Delta7 7/30, EPG 20/17, EPGt 19/18, PEG 18/19,
PEN_a 15/22, PEN_b 18/19; solo Delta7 separa (inhibidor casi suficiente, no necesario) y el contraste de
PEN_b de la tarea antigua (20/32 vs 6/32) desaparece. Concordancia con el veredicto de la tarea antigua:
23 integran en ambas, 5 solo en la antigua, 14 solo en la nueva, 22 en ninguna. Límites: 3 semillas
(con la semilla 0 sola eran 41/64 y puesto 37, así que las cifras se mueven); el umbral de pérdida es
laxo (los barajados por neurona dieron 0.35-0.46) y discrimina la pendiente; descriptivos por tipo sin
test; hiperparámetros elegidos con la real y signo por tipo de origen, como en la sección 6.
  **Análisis exploratorios posteriores a ver los resultados** (`results/types_analysis_v3.json`,
  bloque `exploratory_post_hoc`): (i) el puesto de la real es 37, 36 y 37 por semilla individual
  (30 por la media de las 3); (ii) el recuento depende de la pendiente exigida: 37 patrones con
  > 0.5 o > 0.7, 32 con > 0.8, 20 con > 0.9, y con > 0.8 la real no integra en ninguna semilla; (iii)
  condicionado a Delta7 inhibidor (32 patrones), 30 integran y la real es el puesto 28 de 32 por
  pérdida; con Delta7 excitador integran 7 de 32 y el mejor alcanza 0.058, así que Delta7
  inhibidor no es necesario para integrar bien.
- **Trabajo futuro (pendiente):** explicar mecanísticamente por qué la recurrencia PEN↔PEN del NO es necesaria; reajustar hiperparámetros por grafo.
- Atacar la aprendibilidad (currículo sobre T, otro inicio, tasas de aprendizaje
  por grupo de parámetros, más épocas).
- Medir la potencia del test de H1 para señal a nivel de tipo y usar una nula
  sobre patrones por tipo; comparar el aprendiz de 9.160 ganancias libres por arista
  contra un aprendiz de 6 signos por tipo (más comparable a H1), no solo contra los 64
  patrones enumerados con signos fijos.
- Ampliar el circuito (ring neurons, fan-shaped body) y permitir magnitud
  entrenable como análisis complementario; comparar con el modelo nulo que
  preserva el grado (Dhiman, 2026); comparar en particular con Chang, Huang y Lo (2023)
  y su contraste entre inhibición global por Delta7 e inhibición por ring neurons, que
  este modelo excluye.

## 12. Enunciado propuesto (borrador de texto)

> Preguntamos si una red recurrente restringida solo por la topología real del
> núcleo del sistema de dirección de cabeza de *Drosophila*, con el signo
> sináptico como único parámetro de arista, recupera el neurotransmisor real. No
> encontramos evidencia de ello, pero el diseño no permitía encontrarla:
> la tarea de integración de rumbo era resoluble de forma trivial (un
> decodificador constante superaba a todos los modelos), carecía de pista de fase
> y usaba un mapeo angular que la anatomía medida en sinapsis contradice (los
> hemisferios recorren el anillo en sentidos opuestos). Con una tarea anclada y
> ganancias por tipo celular existe una solución que integra el rumbo, pero no
> es específica de la química real: 28 de los 64 patrones de signo por tipo la
> alcanzan por mayoría de 3 semillas (la asignación real, en el puesto 14 de 64
> por held-out medio; p = 0.22), apoyándose en tasas con signo (desviaciones
> respecto a un nivel basal nulo) que hacen el signo del peso poco identificable
> (con la tarea de velocidad corregida la conclusión se mantiene: 37 de 64 patrones por mayoría,
> la real en el puesto 30);
> y un aprendiz de signos genérico no la encuentra.
> Dado que en este subcircuito el neurotransmisor es función exacta del tipo
> celular, la pregunta se reduce a un dato de tipo, y los cálculos de potencia
> por neurona no son aplicables. Proponemos una lista de comprobaciones previas
> (existencia de una solución trivial, pista de fase, solubilidad con los signos
> anotados, independencia de tasas con signo sin nivel basal, estructura del
> neurotransmisor anotado y respuesta proporcional a la velocidad) antes de
> interpretar acuerdos o desacuerdos de signo en modelos restringidos por conectoma.

## Apéndice: correspondencia con el cuaderno y el código

| hallazgo | entrada del cuaderno | código |
|---|---|---|
| polarización / H1 con `sign_reg` | 2026-09-16 a 09-18 | `train.py`, `evaluate.py` |
| control de potencia por neurona (retirado) | 2026-09-19 (2) | `power_control.py` |
| referencia trivial, signos reales, sin ancla | 2026-09-19 (3), (4) | `real_sign_task_check.py`, `real_sign_offset_check.py` |
| fase real desde sinapsis | 2026-09-19 (5) | `extract_eb_angles.py`, `graph_utils.py` |
| tarea anclada y piloto | 2026-09-19 (6) | `task.py`, `train.py` |
| búsqueda de régimen | 2026-09-19 (7) | `regime_search.py` |
| parámetros por tipo, realizabilidad, barajados | 2026-09-19 (8), (10), (11) | `model.py`, `train.py`, `analyze_dose.py` |
| aprendibilidad | 2026-09-19 (9) | `train.py` |
| 64 patrones por tipo | 2026-09-20, 2026-09-20 (2) | `analyze_types.py` |
| validación mecanística; tasas no negativas | 2026-09-20 (3) | `bump_check.py`, `rate_stats.py` |
| 64 patrones sobre la tarea OU corregida | 2026-10-09, 2026-10-09 (2), 2026-10-12 | `analyze_types_v3.py` |

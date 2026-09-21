# Resultados y discusión metodológica (borrador v2) — CX-Net

> Borrador de trabajo del preprint. Fuente primaria de datos y decisiones:
> `docs/lab-notebook.md` (2026-09-16 a 2026-09-20). **Esta versión reescribe la
> anterior** tras las entradas (4)-(13) del cuaderno: varias afirmaciones de la
> v1 quedan retractadas (tabla de la sección 9). Enfoque: informe metodológico
> y negativo. **Estado: exploratorio.** Los análisis de las secciones 6-7 usan una
> corrida por configuración y solo algunos criterios se fijaron antes de ver los
> resultados (entradas (10) y (12) del cuaderno). La revisión crítica interna está en
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
   de neurotransmisor no es especial**: 26 de los 64 patrones de signo por tipo
   integran y la real queda en el puesto 22 (p exacto = 0.34), apoyándose en
   tasas con signo (desviaciones respecto a un nivel basal nulo) que hacen el
   signo del peso poco identificable (sección 6).
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
  115°, 65°, 22°, 335°, 291°, 239°, 194°, 156°): un espejo, con pasos de ≈45° y
  un desfase L/R de ≈22° entre glomérulos homólogos. EPGt (glomérulo 9) queda en
  113-148°, la fase del glomérulo 1.
- Es consistente con lo descrito para el sistema (cada hemisferio muestrea el anillo
  completo a ≈45° con desfase L/R de 22.5°, Hulse et al., 2021, y la proyección alterna
  de las cuñas del EB a los dos hemisferios del PB viene de trabajos anteriores) y **no
  es un hallazgo anatómico nuevo**: la aportación es una medición reproducible en
  MaleCNS y la corrección de nuestros mapeos previos, que colocaban L y R en mitades
  del círculo (el original) o suponían el mismo sentido en ambos hemisferios (el
  «intercalado» descartado el 2026-09-16).
- Validación con los PEN (no usados para construir el mapa): los PEN de L y R
  se desplazan en sentidos opuestos con consistencia perfecta (≈ −6° y +5.5°;
  magnitud pequeña, probablemente por sinapsis EPG↔PEN recíprocas en el EB que
  el grafo no distingue de las del PB).
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
con el mismo protocolo (una corrida por patrón, semilla 0):

- **26 de 64 patrones integran.** La real (held-out 0.108, pendiente 0.82) es
  el **puesto 22 de 64; p exacto = 22/64 = 0.344.** Patrones biológicamente
  absurdos la superan (p. ej. "solo EPG inhibe": 0.063; "EPG+EPGt inhiben":
  0.063).
- Fracción de patrones que integran según el signo del tipo (excitador /
  inhibidor, de 32 cada uno): Delta7 7 / 19; EPG 10 / 16; EPGt 14 / 12; PEG 12 /
  14; PEN_a 14 / 12; PEN_b 20 / 6. Descriptivo, sin test. Delta7 inhibidor
  ayuda pero no es necesario ni suficiente.
- Los 12 barajados fallan porque rompen la coherencia por tipo, no porque la
  química real sea especial: **la lectura de "suficiencia funcional" de la v1 se
  retracta.**

**Mecanismo probable.** Con `tanh` las tasas van en (−1, 1) y el nivel basal es 0: se
interpretan como desviaciones respecto a una actividad basal. En la red que integra con
la asignación real, Delta7 está por debajo de la basal el 89% del tiempo (fracción > 0 =
0.11; media −0.17): una neurona inhibidora que baja su actividad desinhibe a sus dianas,
con efecto neto excitador. Es una lectura legítima si la neurona real tuviera una
actividad basal tónica suficiente, pero el modelo ni la representa ni la limita, y hace
que el signo del peso deje de ser identificable a partir de la función: muchos patrones
de signo se vuelven funcionalmente equivalentes (comprobación descriptiva sobre 5
ensayos; la simetría de gauge no se ha demostrado formalmente). Con el patrón sin
ninguna inhibición (que no integra), Delta7 tiene fracción > 0 = 1.00. Por tanto la
solución con la asignación real no debe leerse como una solución mecanísticamente
biológica.

**Validación mecanística (descriptiva; `bump_check.py`).** Con velocidad constante
dentro del rango de entrenamiento (±0.01 a ±0.08 por paso durante 90 pasos), la red con
la asignación real desplaza el bump en el sentido correcto, de forma antisimétrica y
aproximadamente proporcional (ganancia 0.54 respecto a un integrador ideal de 1; R² =
0.84), con saturación para desplazamientos grandes (esperado ±7.2 rad, decodificado −2.8
/ +3.3). Otros patrones que integran dan ganancia 0.75-0.76 (R² 0.92-0.93). La red sin
inhibición no responde (desplazamiento ≈ −0.2 rad a todas las velocidades). La
localización del bump (|Σ r e^{iθ}| / Σ|r| sobre EPG/EPGt) es 0.62 en la solución con la
asignación real y 0.34-0.37 en «solo EPG inhibe» y «EPG+EPGt inhiben». Es decir: las
redes que «integran» implementan una integración compresiva (ganancia < 1, saturante),
no un integrador perfecto; y con una velocidad constante grande (±0.3 durante 180
pasos) ningún modelo desplaza el bump. La pendiente de las secciones anteriores mide la
integración en el rango de entrenamiento, no un integrador general. Se comprobó que no
es un artefacto del decodificador.

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
banda de actividad y la fase decodificada en el sentido correcto, más despacio que el
ideal en la asignación real (ganancia 0.54) y algo por debajo en los otros dos (0.75-0.76);
el que no integra no se mueve con ninguna velocidad. La memoria de una fase arbitraria
(8 fases de pista, velocidad 0) no discrimina: los cuatro modelos, incluido el que no
integra, sostienen la fase con error circular medio de 0.06-0.15 rad a t = 60 y 0.18-0.40
rad a t = 199. Lo que separa a los integradores del resto es la respuesta a la
velocidad y la actividad graduada, no la retención de fase. Salvedades: una sola corrida
por modelo, resolución angular gruesa (50 neuronas), y no se comparó con el perfil de
un modelo de anillo construido a mano.

**Tasas no negativas (sigmoide).** Con `activation="sigmoid"` (tasa basal 0.5), mismos
hiperparámetros y protocolo (600 épocas), los signos reales dan held-out 0.303
(`ring_sign=−1`, pendiente 0.33) y 0.470 (`+1`, pendiente 0.01), y 4 barajados
0.536-0.955: los reales quedan por delante de los barajados, pero ninguno cumple el
criterio de «integra». Con esos hiperparámetros la comprobación de realizabilidad no se
cumple, así que no se enumeraron los 64 patrones; haría falta una búsqueda de régimen
propia. El resultado es no concluyente.

## 7. Aprendibilidad

Con signos por arista y parámetros por tipo entrenados a la vez desde un inicio
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

## 8. Interpretación

- **H1 tal como se formuló no está ni confirmada ni refutada**: el aprendiz no
  llega a una solución funcional, y la solución funcional existente no depende
  de la química real. El nulo de la sección 3 no informa sobre la biología.
- **Un bit conocido.** Con el neurotransmisor función del tipo, el contraste de fondo es
  «Delta7 inhibe» (conocido) frente a las otras 63 combinaciones. Que la tarea no lo
  seleccione, con este modelo, dice que la tarea y el modelo no imponen esa
  restricción, no que la biología no la imponga.
- **Sesgo de selección a favor de la real.** Los hiperparámetros dinámicos se eligieron
  en zonas donde la asignación real funcionaba; eso favorecería a la real frente al resto
  de patrones. Que aun así quede en el puesto 22 refuerza el nulo en vez de debilitarlo.
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
2021); los modelos de anillo atractor basados en el conectoma del sistema de dirección de
cabeza, entre ellos Kakaria y de Bivort (2017), Turner-Evans et al. (2020) y Pisokas et
al. (2020), que obtienen dinámica de bump con parámetros ajustados y una inhibición de
Delta7 específica; modelos entrenados con restricción de conectoma (Lappalainen et al.,
2024) y la sensibilidad topológica de esos modelos (Dhiman, 2026); la predicción de
neurotransmisor (Eckstein et al., 2024); y el conectoma MaleCNS (Berg et al., 2026). La
comparación de nuestra dinámica con la de esos modelos ajustados a mano (¿qué parámetros
e inhibición requieren?) es la vía más directa para entender por qué el modelo original
no integra.

## 9. Retractaciones respecto a la v1

| afirmación de la v1 | estado |
|---|---|
| "La red aprende la tarea mejor que el azar (0.6 vs. 0.85)" | Retractada: referencia errónea; los modelos son 3-4× peores que un decodificador constante (sección 4). |
| "Control de potencia: el test detecta ≥ +3.3 pp con ≥ 80%" | Retractada: siembra por neurona/arista, no aplicable a una señal estructurada por tipo (sección 3). Solo se mantiene la calibración del falso positivo. |
| "Signos reales no superan a barajados: 1.006 vs 0.983" (primera lectura) | Superada: prueba injusta sin ancla; ver sección 4. |
| "El mapeo intercalado es el conforme a la anatomía" | Corregida: L y R recorren el anillo en sentidos opuestos (sección 5). |
| "La química real integra y las barajadas no (suficiencia funcional)" | Retractada: 26/64 patrones por tipo integran; la real, puesto 22 (sección 6). |
| "~440 evaluaciones de régimen" | Corregida: 588. |
| «Tasas negativas no fisiológicas» (versión intermedia) | Matizada: son desviaciones respecto a un nivel basal nulo; el problema es la falta de identificabilidad del signo (sección 6). |

## 10. Limitaciones

- **Una corrida por patrón (semilla 0)** en la enumeración de 64: un "no
  integra" puede ser mala suerte de optimización; las réplicas de la asignación
  real (0.106-0.124) sugieren estabilidad, pero no se replicaron los demás
  patrones. Un solo `ring_sign` (+1) en las corridas de refuerzo y de
  enumeración.
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
- **Integración compresiva**: ganancia 0.54-0.76 y saturación; no se compararon las
  redes con integradores ajustados a mano ni se caracterizó el perfil del bump.
- **Entorno**: con torch 2.4.1 (Python del sistema) el gradiente de
  `atan2(0,0)` es `nan`; el entorno del proyecto (`.venv`, torch 2.14) da 0. Los
  entrenamientos por defecto deben ejecutarse con el `.venv`
  (reproducción exacta verificada). Los dos pilotos de la tarea anclada se
  ejecutaron con el Python del sistema (la pista evita el caso `(0,0)`).

## 11. Trabajo futuro

- Buscar un régimen dinámico donde la asignación real integre con tasas no negativas
  (sigmoide, rectificada estable o desviaciones con nivel basal explícito) y solo
  entonces repetir la enumeración de 64 patrones; con los hiperparámetros actuales la
  sigmoide no lo logra.
- Validar el mapeo de fase aplicando el mismo procedimiento a hemibrain y comparándolo
  con las tablas publicadas; comparar la dinámica con modelos de anillo ajustados a mano.
- Replicar cada patrón con varias semillas y ambos sentidos de giro.
- Atacar la aprendibilidad (currículo sobre T, otro inicio, tasas de aprendizaje
  por grupo de parámetros, más épocas).
- Medir la potencia del test de H1 para señal a nivel de tipo y usar una nula
  sobre patrones por tipo.
- Ampliar el circuito (ring neurons, fan-shaped body) y permitir magnitud
  entrenable como análisis complementario; comparar con el modelo nulo que
  preserva el grado (Dhiman, 2026).

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
> es específica de la química real: 26 de los 64 patrones de signo por tipo la
> alcanzan (la asignación real, en el puesto 22; p = 0.34), apoyándose en tasas
> con signo (desviaciones respecto a un nivel basal nulo) que hacen el signo del peso
> poco identificable; y un aprendiz de signos genérico no la encuentra.
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

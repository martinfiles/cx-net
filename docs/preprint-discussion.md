# Discusión metodológica (borrador) — CX-Net

> Borrador de trabajo para la sección de discusión del preprint. Fuente:
> `docs/lab-notebook.md` (registro completo, 2026-09-16 a 2026-09-19). Este
> documento reorganiza esa cronología en una narrativa argumentativa; el
> cuaderno sigue siendo la fuente primaria de datos y decisiones.

## 1. Resultado central

Entrenamos una red recurrente cuya única restricción estructural es la
topología sináptica real del núcleo del sistema de dirección de cabeza del
complejo central de *Drosophila melanogaster* (152 neuronas, 9.160 aristas;
`EPG`, `EPGt`, `PEN_a(PEN1)`, `PEN_b(PEN2)`, `PEG`, `Delta7`; male-cns:v1.0).
La magnitud de cada peso queda fija al número real de sinapsis; el signo
(excitador/inhibidor) es el único parámetro libre, entrenado por descenso de
gradiente sobre una tarea de integración de rumbo, sin acceso nunca al
neurotransmisor real de cada neurona.

La red reduce de forma consistente el error de la tarea respecto a una fase
aleatoria (pérdida held-out ≈0.6 frente a ≈1.0), pero no la resuelve: un
decodificador constante en 0 obtiene 0.18, así que los modelos entrenados son
unas 3-4 veces peores que la solución trivial (ver sección 5). Además,
**el signo sináptico
aprendido no coincide con el neurotransmisor real anotado más de lo
esperable por azar** (test de permutación de etiqueta de neurotransmisor,
2000 permutaciones). De todas las evaluaciones de H1 realizadas a lo largo
del proyecto, ninguna sobrevive como evidencia reproducible a favor de H1.

Este resultado negativo no depende de una sola corrida: se llegó a él tras
descartar sistemáticamente, con metodología multi-semilla, la explicación
más obvia, que la falta de señal se debiera simplemente a una potencia
estadística insuficiente por polarización de signo débil. Un control
positivo de potencia (sección 2) acota además el resultado: el test habría
detectado un acuerdo de +3.3 pp o más con ≥80% de probabilidad, pero no
efectos menores.

## 2. El problema de potencia estadística y cómo se resolvió

Un test de acuerdo de signo solo es informativo si la red se ha
comprometido con un signo por arista (`|tanh(sign_param)|` cercano a 1).
Las primeras evaluaciones de H1 (2026-09-16/17, ver cuaderno, entradas
2026-09-16 (7) y 2026-09-17 (2)/(5)) se hicieron sobre modelos con
polarización débil (`mean_abs_sign` entre 0.31 y 0.36, muy por debajo del
umbral de referencia de 0.5 usado en este proyecto). El resultado (p>0.9)
no era concluyente: un modelo que apenas decidió sus signos no permite
distinguir entre "H1 es falsa" y "el test no tuvo ocasión de detectar
nada".

Se probaron cuatro vías distintas para forzar más compromiso de signo, cada
una con protocolo multi-semilla (3 a 8 semillas, set de validación
held-out fijo, comparación por solapamiento entre grupos) para evitar
repetir el error metodológico temprano del proyecto: una sola corrida por
configuración (entrada 2026-09-16 (11) del cuaderno).

| Vía | Mecanismo | Mejor `mean_abs_sign` | ¿Sin solapamiento vs. baseline? |
|---|---|---|---|
| Hiperparámetros de dinámica (`tau`, `recurrent_gain`, momentum, `trials_per_step`) | Optimización | ~0.31-0.35, sin diferencia consistente | No aplica: no hubo efecto |
| `hold_prob` (rediseño de tarea: tramos de quietud forzada, memoria sin entrada) | Diseño de tarea | 0.349-0.364 | Sí, pero insuficiente para cruzar el umbral |
| `hold_prob` + `perturb_amp` (rediseño de tarea: ruido de entrada a filtrar) | Diseño de tarea | 0.465 ± 0.163 (bimodal) | No, solapa con `hold_prob` solo |
| `hold_prob` + `sign_reg` (regularización directa sobre el parámetro de signo) | Optimización, independiente de la tarea | **0.687 ± 0.123** | **Sí: `mean_abs_sign` de las 8 semillas queda por encima del máximo del baseline** (7 de 8 cruzan además el umbral de referencia) |

Solo las dos últimas vías modifican de forma significativa la
polarización, y solo `sign_reg` (penalización `1 - tanh(sign_param)^2`,
que no favorece ningún signo en particular, solo penaliza la indecisión)
la resuelve de forma limpia y reproducible: 7 de 8 semillas cruzan el
umbral de referencia (`frac_polarized_gt_0.9 > 10%`), con un coste
moderado y desigual en el desempeño de la tarea (`held_out_loss` sube de
0.603±0.009 a 0.631±0.053, aunque la mayoría de las semillas quedan
indistinguibles del baseline).

### Control positivo de potencia del test (medido)

Que la red esté polarizada es condición necesaria, no suficiente, para que
el test sea informativo. Para medir la potencia en vez de inferirla
(`src/cx_net/power_control.py`, `data/interim/power_control.json`) se
sembró señal conocida sobre los signos reales de los 8 modelos `sign_reg`:
una fracción q de las neuronas de origen (o, en la variante de aristas
independientes, de las aristas) pasa a tener el signo de la "verdad", y el
resto conserva el signo aprendido. Se aplicó el mismo test que en H1 (2000
permutaciones, una cola, alfa=0.05), 200 repeticiones por modelo y valor
de q. Como "verdad" se usó una asignación de etiquetas barajada, sorteada
de nuevo en cada repetición (`decoy`), de modo que q=0 mide el falso
positivo y no hereda la señal residual de la semilla 1. Con las etiquetas
reales (`real`) la potencia es similar (hasta ≈0.07 mayor con q bajo, por la
señal residual de la semilla 1).

| q (neuronas sembradas) | acuerdo medio | potencia (neuronas) | potencia (aristas indep.) |
|---|---|---|---|
| 0 (falso positivo) | 0.498 | 0.052 | 0.049 |
| 0.01 | 0.503 | 0.11 | 0.19 |
| 0.02 | 0.508 | 0.26 | 0.47 |
| 0.03 | 0.513 | 0.43 | 0.74 |
| 0.05 | 0.523 | 0.69 | 0.97 |
| 0.10 | 0.548 | 0.98 | 1.00 |
| 0.20 | 0.598 | 1.00 | 1.00 |

El test está calibrado (falso positivo ≈5%). Con siembra a nivel de
neurona, que es la variante realista porque el neurotransmisor es una
propiedad de la neurona y la permutación baraja a ese nivel, la potencia
alcanza ≈50% con un acuerdo de ≈0.516 (+1.6 pp sobre el azar) y ≈80% con
≈0.533 (+3.3 pp); un acuerdo de 0.55 o más se detecta prácticamente
siempre. Los acuerdos observados en los modelos reales (0.476-0.519, media
0.497) quedan todos por debajo del punto de 80% de potencia y solo la semilla
1 supera el de 50%.

Esto sustituye la inferencia por una medición y acota el resultado nulo:
**un acuerdo de signo de +3.3 pp o más sobre el azar se habría detectado
con ≥80% de probabilidad; el diseño no excluye efectos menores** (del
orden de +1-2 pp sobre el azar; la única desviación nominal observada, la
semilla 1 con acuerdo 0.519, cae justo en la zona de ≈50% de potencia). Limitaciones del control: la señal sembrada es un modelo
idealizado (neuronas enteras con el signo correcto, sin ruido intermedio)
y mide la potencia del test de permutación, no la capacidad del
entrenamiento de recuperar signos; esa segunda pregunta requeriría entrenar
sobre una red con signos conocidos (variante no realizada).

## 3. Con la polarización resuelta, H1 sigue sin evidencia

Este es el punto argumentativo central de la discusión: con `sign_reg`, la
falta de polarización deja de ser una explicación viable para la ausencia
de señal. Sobre las 8 semillas de `hold_prob=0.3` + `sign_reg=0.05` (7 de
ellas por encima del umbral de referencia), el acuerdo de signo con el
neurotransmisor real es prácticamente el del azar en todas:

| semilla | acuerdo observado | media nula | z | p (una cola) |
|---|---|---|---|---|
| 0 | 0.4965 | 0.4978 | −0.28 | 0.618 |
| 1 | 0.5193 | 0.5071 | +2.49 | 0.0055 |
| 2 | 0.5029 | 0.4999 | +0.57 | 0.294 |
| 3 | 0.5038 | 0.4987 | +1.05 | 0.152 |
| 4 | 0.4957 | 0.4987 | −0.44 | 0.686 |
| 5 | 0.4758 | 0.4938 | −3.16 | 0.999 |
| 6 | 0.4855 | 0.4929 | −1.33 | 0.913 |
| 7 | 0.4932 | 0.4963 | −0.65 | 0.742 |

Tres puntos que el resumen "solo 1 de 8 sale significativo" oculta y que
conviene decir de forma explícita:

- **La semilla 1 (p=0.0055) no es descartable por simple recuento de
  falsos positivos al 5%.** Bajo la nula, la probabilidad de que el mínimo de
  8 p-valores baje de 0.0055 es ≈4.3%, y 0.0055 queda por debajo del umbral
  de Bonferroni (0.05/8=0.00625). Lo que la hace no concluyente es que su
  efecto es diminuto (acuerdo 0.519 frente a 0.507, +1.2 puntos porcentuales),
  que no se replica en las otras semillas y que la semilla 5 muestra la
  desviación opuesta, de mayor magnitud (z=−3.16), que un test de una cola
  (p=0.999) no señala. Las desviaciones a ambos lados son coherentes con
  ruido de optimización, no con un acuerdo sistemático. El control de
  potencia (sección 2) explica además por qué un efecto de este tamaño no
  es decisivo en ningún sentido: se detecta solo la mitad de las veces si
  existe, de modo que verlo en 1 de 8 semillas es compatible tanto con
  ruido como con un efecto pequeño real.
- **Agregado, no hay señal:** z medio −0.22; Stouffer z=−0.62
  (p=0.73, una cola); Fisher p=0.26. Además, los 8 modelos se evalúan contra
  las mismas etiquetas reales, así que no son 8 pruebas independientes de
  H1 sino 8 muestras del mismo procedimiento de entrenamiento.
- **El mismo patrón, un resultado significativo aislado que no se replica,
  ya había aparecido con `perturb_amp`** (1 de 8 semillas, p=0.0005 en el
  mínimo posible con 2000 permutaciones; entrada 2026-09-18 del cuaderno). Ese
  resultado se retractó por un criterio distinto (la semilla polarizaba por
  azar de optimización, no por efecto de la condición), no por un análisis de
  comparaciones múltiples, y debe reportarse así.

La evidencia decisiva es solo la de `sign_reg`. `hold_prob` y
`hold_prob`+`perturb_amp` no lograron polarizar de forma suficiente y
reproducible (esta última solo en 1 de 8 semillas), de modo que no
constituyen pruebas independientes con potencia adecuada de H1; son
intentos fallidos de alcanzarla. Presentarlos como "dos mecanismos
ortogonales que convergen" sobreestima el peso de la convergencia: lo que
convergen es la ausencia de señal, pero solo uno de los dos mecanismos
llega a tener potencia para detectarla.

## 4. Interpretación

Los datos son compatibles con, al menos, tres lecturas distintas, que este
proyecto no puede distinguir entre sí con el diseño actual:

1. **La topología por sí sola no codifica información suficiente sobre
   identidad química** como para que un descenso de gradiente agnóstico a
   la tarea pueda recuperarla. El signo sináptico real podría depender de
   más contexto del que capta este subcircuito acotado (152 neuronas
   núcleo, sin los ~30 subtipos de ring neurons ni los
   PFN/PFL/hDelta/vDelta del fan-shaped body completo).
2. **La tarea conductual elegida (integración de rumbo) no impone las
   mismas restricciones funcionales** que dieron forma a la asignación
   real de neurotransmisor a lo largo de la evolución. Por bien resuelta
   que esté, una sola tarea es una ventana estrecha sobre las demandas
   funcionales reales del circuito. **Dos comprobaciones sin entrenamiento
   apoyan que la tarea no discrimina la química** (`real_sign_task_check.py`,
   `real_sign_offset_check.py`). (i) Con la pérdida original, los signos
   reales fijos dan 1.006 frente a 0.983 ± 0.103 de 500 asignaciones de
   neurotransmisor barajadas (p=0.58). Esa no es una prueba justa: la
   pérdida exige el rumbo absoluto sin dar pista de dónde está el 0, así que
   ni un integrador perfecto con fase inicial arbitraria baja de ≈1.0.
   (ii) Con una pérdida invariante al desfase constante y tres mapeos de
   `ring_angle` (el actual y dos intercalados, conformes a Hulse et al.
   2021), los signos reales tampoco superan a los barajados (p=0.98, 0.77 y
   0.55; 150 asignaciones por mapeo) y quedan por encima del suelo trivial
   (0.06, decodificador constante): 0.115-0.459. Es decir, en este modelo
   los signos reales no producen una red que integre el rumbo, y el 5%
   inferior de las asignaciones barajadas (≈0.067) es indistinguible de una
   red estática. Matiz: esto no prueba que la biología no resuelva la tarea;
   solo que el modelo simplificado (sin ring neurons ni fan-shaped body,
   entrada de velocidad angular inyectada a mano, `recurrent_gain` y
   normalización por nodo elegidos por nosotros) no la reproduce con los
   signos reales. Los modelos entrenados se evaluaron solo con el mapeo
   actual.
3. **Limitaciones de la arquitectura**: la magnitud de cada peso queda fija
   al conteo de sinapsis (una decisión deliberada, para que el signo sea
   la única variable libre y H1 sea una prueba limpia, ver `model.py`).
   Pero esa misma decisión le quita a la red la posibilidad de compensar
   un signo equivocado con la magnitud, que es precisamente el mecanismo
   que permitiría distinguir una arista donde el signo importa de otra
   donde no.

Ninguna de las tres lecturas implica que H1 sea falsa en general. Lo que
sí implican es que **este diseño experimental concreto solo excluye
efectos de acuerdo de signo de +3.3 pp o más (potencia ≥80%, sección 2) y
no puede refutar ni confirmar de forma concluyente efectos menores**,
incluso después de resolver el cuello de botella de la polarización. Más
importante aún: como los modelos no resuelven la tarea mejor que una
solución trivial (sección 5), es dudoso que los signos aprendidos reflejen
alguna restricción funcional; el resultado nulo informa sobre este montaje
(tarea y decodificador) más que sobre la biología. Ese es el hallazgo
metodológico que se reporta.

## 5. Limitaciones explícitas

- **`ring_angle` inconsistente con la anatomía**: el mapeo del proyecto
  (`actual`) coloca los 8 glomérulos de L en media circunferencia y los de R
  en la otra media. Hulse et al. (2021, texto de EPG y Fig. 16) describen que
  cada hemisferio del PB muestrea el anillo completo a ≈45° (8 glomérulos por
  lado), con un desfase L/R de 22.5°, y que EPGt (glomérulo 9) equivale en fase
  al glomérulo 1. El mapeo intercalado que se descartó el 2026-09-16 (entrada
  10) es el conforme al artículo; se revirtió por su efecto sobre la
  polarización, no por un criterio anatómico. La tabla exacta (Fig. 10) no
  pudo extraerse del texto y el sentido de giro no está verificado. Todos los
  modelos entrenados de este trabajo usan el mapeo `actual` y no se
  reentrenó con el corregido. El test de permutación de H1 no usa
  `ring_angle`, pero `task.py` sí lo usa para el objetivo de decodificación
  durante el entrenamiento, así que un mapeo erróneo cambia la tarea que la
  red aprende y, con ella, los signos aprendidos: es una fuente potencial de
  sesgo sobre H1.
- **Alcance del subcircuito**: 152 neuronas núcleo del sistema de
  dirección de cabeza, sin las ring neurons de entrada visual ni el
  fan-shaped body completo. Un circuito más amplio podría comportarse de
  forma distinta.
- **Una sola tarea conductual**: integración de rumbo, con dos variantes
  de dificultad (`hold_prob`, `perturb_amp`). No se probaron tareas
  cualitativamente distintas (navegación hacia una meta, integración
  multisensorial) por restricción de alcance y tiempo del proyecto.
- **Desglose H2 con n bajo por tipo celular**: 152 neuronas repartidas en
  6 tipos (`Delta7`=42, `EPGt`=4). Los tipos con menos neuronas tienen
  pocas aristas de origen, así que cualquier señal, o ausencia de señal,
  en ellos debe leerse con esa salvedad.
- **La red no resuelve la tarea**: la pérdida held-out (≈0.6) es 3-4 veces
  peor que la de un decodificador constante en 0 (0.18; el rumbo tiene
  desviación final ≈1 rad, así que no moverse ya da pérdida baja). La mejora
  "sobre el azar" se mide contra una fase aleatoria (≈1.0), una referencia
  que no es pertinente. Con la pérdida invariante al desfase, los modelos
  entrenados dan 0.08-0.36 frente a 0.06 del decodificador constante. No se
  ha demostrado que la red implemente un integrador de rumbo, biológico o no.
  Si la solución no se parece al mecanismo real, no hay razón para esperar
  que sus signos coincidan con los reales.
- **El modelo con signos reales tampoco integra el rumbo** (ver sección 4,
  lectura 2): con signos reales fijos la pérdida invariante al desfase es
  0.115-0.459 según el mapeo, frente a 0.06 del decodificador constante, y no
  mejor que con signos barajados. Hasta que un modelo con signos reales
  resuelva la tarea, esta no es un banco de pruebas válido para preguntar si
  el entrenamiento recupera la química. Además, forzar a ±1 los signos de
  los modelos entrenados degrada su pérdida original (0.58-0.72 → 0.70-0.79):
  la solución aprendida usa valores graduados de `tanh(sign_param)`, es decir,
  cierta magnitud efectiva, así que "signo libre, magnitud fija" no se cumple
  estrictamente en la solución y el acuerdo de signo (`sign(sign_param)`)
  descarta información que la red usa.
- **Control positivo de potencia parcial**: se midió la potencia del test
  de permutación sembrando señal conocida sobre los signos reales (sección
  2): ≈80% de detección para +3.3 pp de acuerdo, sin poder para efectos de
  +1-2 pp. Lo que no se hizo es el control más exigente: entrenar sobre una
  red con signos conocidos para comprobar que el *entrenamiento* recupera
  el signo cuando existe. Sin él, un resultado nulo también es compatible
  con que el entrenamiento no recupere señal que sí está presente.
- **Grados de libertad del investigador**: ~30 configuraciones y umbrales
  de referencia (`mean_abs_sign>0.5`, `frac_polarized_gt_0.9>10%`) fijados
  sin análisis previo de potencia. Es coherente con un resultado nulo, pero
  hace que ningún p-valor aislado deba leerse como confirmatorio.
- **`sign_reg` es una intervención posterior al diseño original**: se
  añadió específicamente para resolver la duda de potencia estadística
  tras observar el techo de polarización. Es un control metodológico
  válido para esa pregunta puntual, pero no debe presentarse como parte
  del diseño experimental original.

## 6. Qué no se hizo (fuera de alcance, candidatos para trabajo futuro)

- Escalar a un subcircuito CX más amplio o al connectoma completo de
  MaleCNS v1.0.
- Probar tareas conductuales cualitativamente distintas, o varias tareas
  simultáneas, más allá de las dos variantes de dificultad probadas.
- Permitir magnitud entrenable. Esto rompería la limpieza de H1 tal como
  está planteada, pero podría usarse como análisis complementario para
  distinguir aristas sensibles al signo de aristas donde el signo no
  importa.
- Comparar directamente contra el modelo nulo que preserva el grado
  topológico, usado por Dhiman (2026). Aquí se usó permutación de la
  etiqueta de neurotransmisor, que responde una pregunta distinta (acuerdo
  de signo, no velocidad de aprendizaje).

## 7. Enunciado propuesto para la sección de discusión (borrador de texto)

> Entrenar una red cuya única variable libre es el signo sináptico, sobre
> la topología real de un circuito de navegación bien caracterizado,
> reduce el error en una tarea de integración de rumbo respecto a una fase
> aleatoria, aunque sin alcanzar el de una solución trivial, y sin que el signo aprendido reproduzca la identidad química real
> más allá del azar. Un primer intento de rediseñar la tarea no consiguió
> que la red se comprometiera con un signo por arista, lo que dejaba abierta
> la duda de si el test tenía potencia. Una regularización aplicada
> directamente sobre el parámetro de signo resolvió ese cuello de botella
> (polarización por encima del umbral de referencia en 7 de 8 semillas)
> sin alterar el resultado: el acuerdo con el neurotransmisor real
> permaneció indistinguible del esperado por azar, tanto por semilla
> (una semilla nominalmente significativa, con un efecto de +1.2 puntos
> porcentuales y otra con desviación opuesta de mayor magnitud) como en
> agregado (Stouffer z=−0.62). En este subcircuito y con esta tarea, por
> tanto, no encontramos evidencia de que la topología sináptica por sí
> sola baste para que un entrenamiento por gradiente recupere la química
> real de las sinapsis. Un control positivo de potencia (señal
> sembrada sobre los signos reales) muestra que el test detecta con ≥80% de
> probabilidad un acuerdo de +3.3 puntos porcentuales o más sobre el azar,
> pero no efectos menores. Dado que la red no resuelve la tarea mejor que una
> solución trivial y que no se comprobó que el entrenamiento recupere signos conocidos, este
> resultado no permite descartar que la señal exista, con menor magnitud, o
> bajo una tarea o una arquitectura distintas.

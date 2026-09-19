# Cuaderno de laboratorio — CX-Net

Registro cronológico de decisiones, resultados y pendientes. Base para redactar
la memoria final (introducción, metodología y discusión de resultados).

## 2026-09-16 — Definición del alcance y pivote de hipótesis

**Contexto de partida.** El 3 de septiembre de 2026, Google Research y Janelia
(HHMI) publicaron el connectoma completo del sistema nervioso de un macho de
*Drosophila melanogaster* (MaleCNS v1.0): 166.000 neuronas, 125 millones de
sinapsis, con neurotransmisor predicho por neurona.

**Hipótesis inicial (descartada).** ¿Aporta la topología sináptica real del
complejo central (circuito de navegación/heading) una ventaja de aprendizaje
frente a una topología de control aleatoria, entrenando ambas por descenso de
gradiente al estilo flyvis (Lappalainen et al., 2024, *Nature*)?

**Motivo del descarte.** Revisión de literatura (bioRxiv/arXiv, semanas previas
al 2026-09-15) encontró:

- Dhiman (2026, arXiv:2604.04033) ya responde esta pregunta con controles
  estadísticos rigurosos (misma inicialización, modelo nulo que preserva el
  grado) y encuentra que la ventaja de la topología real **desaparece**.
- La comparación macho/hembra (cortejo, agresión) para explicar diferencias de
  cableado es el paper compañero oficial del propio dataset (Cell, 2026-09-03)
  — no disponible como aportación externa.
- Modelado general de navegación/heading con redes entrenadas está saturado
  (FLYNN, varios preprints de orientación, repos hobbyistas).

**Hipótesis final (H1/H2).** Ocultando el neurotransmisor real durante el
entrenamiento y dejando libre el signo de cada peso sináptico, ¿converge la
red por sí sola hacia la química real anotada en el dataset, por encima de un
modelo nulo que preserva el grado? (H1, principal) ¿Varía ese acuerdo por
neuropilo/tipo celular? (H2, análisis secundario).

**Decisión de alcance.** Piloto sobre hemibrain (API de neuprint madura y
estable) antes de escalar a MaleCNS v1.0 (API CAVE, más nueva y con más
fricción de acceso). Subcircuito: PB + EB + FB + NO (~200-300 neuronas).

**Pendiente inmediato.** Verificar en vivo el nombre exacto de la columna de
neurotransmisor predicho en el esquema de neuprint instalado (`extract_graph.py`
tiene varios candidatos marcados con TODO) antes de la primera ejecución real,
que requiere token personal de neuprint.janelia.org.

## 2026-09-16 (2) — Primera ejecución de `extract_graph.py`

**Qué se hizo:** cuenta creada en neuprint.janelia.org, token configurado, entorno
virtual instalado (`requirements.txt`), primera ejecución completa del script de
extracción sobre hemibrain:v1.2.1.

**Incidencias resueltas:**
1. `NeuronCriteria(rois=CX_ROIS)` sin `roi_req="any"` exige presencia en
   *todos* los ROIs a la vez (default `"all"`) → solo devolvía 30 neuronas.
   Corregido a `roi_req="any"`.
2. `fetch_adjacencies()` devuelve `(neurons_df, roi_conn_df)` en ese orden —
   el código asumía el orden contrario y asignaba la tabla de conexiones a
   una variable que en realidad contenía neuronas. Confirmado con
   `help(fetch_adjacencies)` y una prueba aislada sobre el ROI EB. Corregido.
3. El hemibrain **no tiene neurotransmisor predicho** en el esquema de
   neuprint (columnas disponibles listadas en el código). Es un dato
   solo disponible en MaleCNS v1.0 vía CAVE. El script ahora lo detecta y
   avisa en vez de fallar; el ground truth real se extraerá en la
   migración a CAVE.

**Resultado:** `data/raw/graph_no_sign.csv` — 470.649 aristas, 3.085 neuronas.

**Hallazgo relevante para el alcance:** `roi_req="any"` es demasiado laxo —
incluye neuronas que solo tocan tangencialmente el CX (fibras de paso), muy
lejos de las ~200-300 neuronas "núcleo" asumidas en la propuesta inicial.

**Siguiente paso:** decidir el criterio de inclusión definitivo antes de la
fase 2 — opciones: (a) lista cerrada de tipos celulares canónicos del CX
(E-PG, P-EN1/2, Δ7, P-FN, según Hulse et al. 2021), o (b) umbral mínimo de
sinapsis dentro de los ROIs del CX. Probablemente (a) es más defendible
porque es exactamente el conjunto que la literatura ya trata como "el
circuito".

## 2026-09-16 (3) — Acotar a tipos celulares canónicos + inventario de hardware

**Por qué:** `roi_req="any"` por sí solo daba 3.085 neuronas / 470.649 aristas —
incluye fibras de paso sin relación funcional con el circuito de rumbo, y no
encaja con el alcance de cómputo previsto (GPU de consumo, 12 semanas
part-time). Se acota a los tipos celulares que la literatura (Hulse et al.
2021; Turner-Evans et al. 2017; Green et al. 2017) reconoce como el núcleo
del sistema de dirección de cabeza (ring attractor): `EPG`, `EPGt`,
`PEN_a(PEN1)`, `PEN_b(PEN2)`, `PEG`, `Delta7`. Nombres de tipo verificados en
vivo contra el esquema real de hemibrain:v1.2.1 (no asumidos de memoria).

Quedan deliberadamente fuera de esta primera iteración: los ~30 subtipos de
ring neurons (ER/ExR, entrada visual al EB) y los PFN/PFL/hDelta/vDelta del
fan-shaped body — pertenecen al CX completo pero no al núcleo de heading, y
son candidatos para una extensión futura si H1/H2 dan resultado.

**Resultado:** `data/raw/graph_no_sign.csv` — 9.722 aristas, 152 neuronas.
Tamaño en línea con lo previsto en la propuesta (~150-300 neuronas).

**Hardware disponible:** GPU local AMD Radeon RX 7700 XT (RDNA3). PyTorch con
ROCm solo tiene soporte oficial en Linux; en Windows la vía sería
`torch-directml` o WSL2 + ROCm. No es bloqueante para el tamaño actual del
grafo (probable que CPU sea suficiente); queda anotado como opción si el
proyecto escala a un subcircuito mayor de MaleCNS más adelante.

## 2026-09-16 (4) — Fase 2: arquitectura, tarea y primer entrenamiento piloto

**Qué se construyó:**
- `graph_utils.py`: carga el grafo de la Fase 1 y estima una posición angular
  (`ring_angle`) por neurona a partir del número de glomérulo parseado del
  campo `instance` real (p. ej. `EPG(PB08)_L3`). **Aviso:** es una
  aproximación secuencial (glomérulo 1-8 por hemisferio -> 16 posiciones
  equiespaciadas), NO la tabla de correspondencia glomérulo-cuña real de la
  literatura (Turner-Evans et al. 2017; Hulse et al. 2021). Pendiente de
  contrastar antes de confiar en resultados cuantitativos de decodificación.
- `model.py`: red recurrente donde la MAGNITUD de cada peso es el número real
  de sinapsis (fijo) y el SIGNO es el único parámetro entrenable (vía
  `tanh(sign_param)`), tal como exige H1.
- `task.py`: tarea de integración de rumbo — velocidad angular inyectada de
  forma asimétrica en PEN_a/PEN_b (L/R con signo opuesto), rumbo decodificado
  como vector poblacional sobre EPG/EPGt, pérdida circular `1 - cos(diff)`.
- `train.py`: bucle de entrenamiento con Adam sobre `sign_param` únicamente.

**Incidencia encontrada y corregida:** la suma de sinapsis entrantes por
neurona en datos reales es enorme (media ~850, máximo ~1493) frente a lo que
`tanh` puede procesar sin saturar (~±3). Sin corregirlo, la red se satura por
completo y el gradiente se anula — el primer entrenamiento no mejoraba de
forma consistente. Solución: normalizar cada peso por el total de sinapsis
entrantes de su neurona destino (preserva las proporciones relativas reales
entre inputs de una neurona, solo reescala la magnitud absoluta) y añadir una
ganancia recurrente global entrenable como hiperparámetro.

**Resultado del piloto (150 épocas, 8 ensayos/paso, hemibrain, sin ground
truth de NT todavía):** pérdida baja de ~0.85 (nivel de azar) a un rango
estable de ~0.5-0.6 (mínimo puntual 0.30). Mejor que azar de forma
consistente, pero sin convergencia limpia — probable necesidad de ajustar
τ (constante de tiempo), la ganancia de entrada externa, o entrenar más
épocas con una tasa de aprendizaje menor.

**Importante — alcance de este resultado:** esto valida que el pipeline
(datos -> arquitectura -> tarea -> entrenamiento) funciona de extremo a
extremo. NO evalúa H1 todavía: el hemibrain no tiene neurotransmisor real
anotado, así que no hay nada con qué comparar el signo aprendido. Ese paso
requiere migrar la extracción a MaleCNS v1.0 vía CAVE.

**Siguiente paso:** (a) ajuste fino de hiperparámetros de la dinámica para
mejorar la convergencia, (b) validar `ring_angle` contra la tabla real de
Hulse et al. (2021), (c) migrar `extract_graph.py` a `caveclient` sobre
MaleCNS v1.0 para obtener el ground truth de neurotransmisor real.

## 2026-09-16 (5) — Chequeo de cordura + scheduler + mejor checkpoint

**Por qué:** el entrenamiento estocástico (150 épocas) no convergía de forma
limpia. Antes de seguir ajustando a ciegas, se hizo el chequeo estándar:
¿puede el modelo sobreajustar UN solo ensayo fijo? Si no, el problema es de
diseño; si sí, es de optimización/variancia entre ensayos.

**Chequeo de cordura (`sanity_check.py`):** sobreajustando un ensayo fijo
(500 pasos, lr 0.05), la pérdida baja de 0.93 a ~0.05-0.10 de forma estable.
Confirma que arquitectura, tarea y pérdida están bien planteadas.

**Diagnóstico del entrenamiento estocástico:** con más pasos de gradiente
(500 épocas, lr 0.05) la pérdida baja bien al principio (mínimo 0.31 hacia
la época 175) pero luego oscila y empeora -- señal de tasa de aprendizaje
demasiado alta para la fase final, agravada por el ruido de usar ensayos
aleatorios distintos en cada paso.

**Corrección aplicada (práctica estándar):** `ReduceLROnPlateau` (reduce la
tasa a la mitad cada vez que la pérdida deja de mejorar 30 épocas seguidas) +
seguimiento del mejor checkpoint visto durante el entrenamiento (no el
último, que es ruidoso).

**Resultado final del piloto:** mejor pérdida = **0.29** (frente a ~0.85 de
nivel-azar), alcanzada hacia la época 200; el modelo guardado (`model_pilot.pt`)
corresponde a ese checkpoint, no al último. Confirma que el pipeline aprende
una representación de rumbo mejor que el azar de forma consistente y
reproducible.

**Siguiente paso:** con el pipeline validado y estable, el paso de mayor
valor ahora es migrar `extract_graph.py` a `caveclient` sobre MaleCNS v1.0
para obtener el ground truth de neurotransmisor real y poder evaluar H1 por
primera vez -- seguir afinando hiperparámetros sobre hemibrain (sin ground
truth) tiene rendimiento decreciente en este punto.

## 2026-09-16 (6) — Migración a male-cns:v1.0: no hacía falta CAVE

**Hallazgo:** antes de montar autenticación CAVE, se comprobó si neuprint ya
servía el MaleCNS directamente. Sí: `male-cns:v1.0` está disponible en
neuprint.janelia.org con el MISMO `NEUPRINT_TOKEN` que hemibrain -- cero
fricción adicional, no hizo falta crear cuenta CAVE. `extract_graph.py` se
parametrizó para aceptar `--dataset` (hemibrain para el piloto, male-cns:v1.0
por defecto para el experimento real), cada uno en su propia subcarpeta de
`data/raw/`.

**Resultado de la extracción real:** mismo conjunto de 152 neuronas núcleo
(EPG 46, Delta7 42, PEN_b 22, PEN_a 20, PEG 18, EPGt 4) -- consistente con
hemibrain, razonable dado que este circuito es muy estereotipado entre
individuos. 9.160 aristas (frente a 9.722 en hemibrain: ligera diferencia
esperable entre dos reconstrucciones de individuos distintos).

**Neurotransmisor real (ground truth, antes oculto):**
- Delta7 -> **glutamato** (inhibidor en este circuito, vía canales de cloro).
- EPG, EPGt, PEG, PEN_a, PEN_b -> **acetilcolina** (excitador).

Coincide exactamente con lo que Turner-Evans et al. (2017) y Green et al.
(2017) ya asumían a mano en sus modelos analíticos (Delta7 inhibidor,
resto excitador) -- buena señal de que el ground truth es sólido.
Consecuencia práctica para H1: como el neurotransmisor es una propiedad de
la neurona (no de la sinapsis individual), el signo esperado de cada arista
se define por el tipo de su neurona de ORIGEN, no por tipo de conexión.

**Siguiente paso:** reentrenar el modelo de signo libre sobre el grafo real
de male-cns:v1.0 (los bodyId no coinciden con hemibrain, hace falta grafo
nuevo) y evaluar H1: % de acuerdo entre signo aprendido y signo esperado por
neurotransmisor, contra un modelo nulo que preserva el grado (Dhiman, 2026).

## 2026-09-16 (7) — Primer intento de H1: resultado prematuro, no válido

**Qué se hizo:** con el checkpoint reentrenado sobre male-cns:v1.0 (mismo
entrenamiento de la Fase 2, best loss 0.2968), se implementó `evaluate.py`:
signo esperado por arista = neurotransmisor real de la neurona de ORIGEN
(acetilcolina -> +1, glutamato -> -1); control estadístico = test de
permutación de qué neurona tiene qué neurotransmisor (2000 permutaciones;
nota: esto es distinto del modelo nulo de Dhiman 2026, que baraja topología
para una pregunta sobre velocidad de aprendizaje -- aquí la pregunta es de
acuerdo de signo, así que el control correcto es permutar la etiqueta de
neurotransmisor, no la topología).

**Resultado bruto:** acuerdo observado 0.506, acuerdo esperado por azar
0.506, p=0.484 -- H1 no soportada.

**Por qué NO se acepta este resultado todavía:** antes de concluir nada se
comprobó cuán "decididos" estaban los signos aprendidos. Solo el 0.01% de
las aristas tenía un signo fuertemente polarizado (|tanh|>0.9) y el 62%
seguía prácticamente sin decidir (|tanh|<0.3) -- es decir, la mayoría de los
signos apenas se habían movido de su inicialización aleatoria. El chequeo de
cordura (sobreajuste de un ensayo, pérdida ~0.02-0.08) sí polariza con
fuerza (71% de media, 21% totalmente decididos). Conclusión: el modelo no
había entrenado lo suficiente como para que el test de H1 fuera informativo
-- un resultado negativo con signos sin decidir es indistinguible del ruido,
no es evidencia real en contra de H1.

**Incidencia en el reentrenamiento largo (2500 épocas):** `ReduceLROnPlateau`
sobre la pérdida cruda de cada época (muy ruidosa, cada época usa ensayos
aleatorios distintos) confundió ruido con estancamiento y bajó la tasa de
aprendizaje a ~1e-8 hacia la época 700 -- las 1800 épocas restantes no
aprendieron nada (best loss quedó en 0.2441, apenas mejor, polarización
igual de baja). Corrección: el scheduler ahora decide sobre una media móvil
exponencial de la pérdida (`ema_alpha=0.05`), no sobre el valor crudo, con
`patience=150` en vez de 30; además se subió `trials_per_step` a 16 para
reducir el ruido de base. Repitiendo el entrenamiento (1500 épocas) con esta
corrección -- resultado pendiente, se registra en la próxima entrada.

## 2026-09-16 (8) — Corte de sesión: estado y próximos pasos

**Por qué se corta aquí:** fin de la jornada (apagado del equipo). El
reentrenamiento largo (1500 épocas, `trials_per_step=16`, scheduler
corregido con EMA) se lanzó en segundo plano pero se detuvo manualmente
antes de terminar -- no se pierde nada irrecuperable porque `train.py` NO
guardaba checkpoints intermedios en ese momento. **Se corrigió esto mismo:**
ahora guarda cada 100 épocas en `data/interim/model_checkpoint_inprogress.pt`
+ `data/interim/training_progress.json` (época actual, mejor pérdida, lr),
y los prints usan `flush=True` (antes no se veía nada en el log de un
proceso en segundo plano hasta que terminaba, por buffering de Python).

**ESTADO EXACTO al cortar:**
- ✅ Fases 1-2 completas y validadas (extracción, arquitectura, chequeo de
  cordura, entrenamiento con checkpoint-mejor + scheduler EMA corregido).
- ✅ Ground truth real de male-cns:v1.0 extraído (`data/raw/malecns/`):
  Delta7=glutamato (inhibidor), resto del núcleo=acetilcolina (excitador).
- ✅ `evaluate.py` implementado y probado (test de permutación de etiqueta
  de neurotransmisor, desglose H2 por tipo celular).
- ❌ **NO hay todavía un modelo bien convergido sobre male-cns:v1.0 con el
  scheduler corregido.** El único checkpoint guardado en
  `data/interim/model_pilot.pt` es el de la corrida ANTERIOR al fix del
  scheduler (best loss 0.2968, pero con señal de signo débil -- NO fiable
  para evaluar H1, ver entrada anterior). El intento de 1500 épocas con la
  corrección se detuvo a medias sin guardar nada útil (se cortó antes de
  llegar a la época 100, primer punto de guardado).

**PRÓXIMOS PASOS (en orden, para retomar mañana):**

1. Activar entorno: `cd cx-net && .venv\Scripts\activate` (Windows) o
   `source .venv/Scripts/activate` (Git Bash).
2. Relanzar el entrenamiento corregido: `python -m src.cx_net.train`
   (1500 épocas, ~20-25 min en CPU -- lanzar con tiempo de sobra, o en
   segundo plano). Vigilar que `ema` en los logs baje de forma sostenida y
   que la tasa de aprendizaje NO caiga a valores absurdos (~1e-8) antes de
   la época 1000 -- si pasa otra vez, subir `patience` todavía más.
3. Tras entrenar, comprobar polarización de signos ANTES de evaluar H1
   (umbral orientativo: `mean_abs_sign` > 0.5 y `frac_polarized_gt_0.9` >
   0.1, comparable al chequeo de cordura). Si sigue bajo, no evaluar H1
   todavía -- seguir ajustando (más épocas, más `trials_per_step`, o revisar
   si `recurrent_gain`/`tau` necesitan cambiar).
4. Solo si la polarización es razonable: `python -m src.cx_net.evaluate` y
   registrar el resultado (observed_agreement, p_value, desglose H2) tal
   cual salga, sea cual sea -- un negativo bien fundamentado sigue siendo
   publicable (ver propuesta original).
5. Pendiente aparte, no bloqueante: validar `ring_angle` (posición angular
   aproximada usada para decodificar el rumbo) contra la tabla real de
   Hulse et al. (2021) antes de confiar en resultados cuantitativos de
   decodificación -- está marcado como aproximación no verificada desde la
   entrada (4).

**Archivos que importan para retomar:** `docs/lab-notebook.md` (este
archivo, léelo entero de arriba a abajo si retomas en otra sesión de
Claude), `src/cx_net/*.py`, `data/raw/malecns/` (datos ya descargados, no
hace falta re-extraer), `data/interim/` (vacío de resultados útiles ahora
mismo, todo lo de ahí es de la corrida descartada).

## 2026-09-16 (9) — Retomando: scheduler EMA corregido funciona, pero polarización sigue por debajo del umbral

**Qué se hizo:** se retomó la sesión, se confirmó el estado exacto descrito en la
entrada (8) (ningún checkpoint útil del intento cortado) y se relanzó el
entrenamiento con el scheduler EMA corregido, en dos configuraciones
sucesivas sobre male-cns:v1.0:

1. `n_epochs=1500, trials_per_step=16` (la config que ya estaba puesta por
   defecto en `train.py`).
2. `n_epochs=2500, trials_per_step=32` (doblar ensayos por paso para reducir
   ruido de gradiente, ya que el LR ahora decae de forma sana pero la EMA
   seguía oscilando ~0.48-0.51 sin bajar limpio).

**Resultado:**

| Corrida | best_loss | mean_abs_sign | frac_polarized_gt_0.9 |
|---|---|---|---|
| Descartada (2500ep/16, scheduler roto, entrada 7) | 0.2441 | 0.256 | 0.0002 |
| 1500ep / 16 trials | 0.3128 | 0.321 | 0.021 |
| 2500ep / 32 trials | 0.3978 | 0.351 | 0.060 |

(Umbral orientativo definido en la entrada (8): `mean_abs_sign > 0.5` y
`frac_polarized_gt_0.9 > 0.1`, comparable al chequeo de cordura de la
entrada (5): ~71% / 21%.)

**Lectura:** el scheduler EMA ya no colapsa el LR de forma prematura (bug de
la entrada 7 corregido y confirmado). Doblar `trials_per_step` mejora la
polarización de forma clara (~3x en `frac_polarized_gt_0.9`) aunque empeore
el `best_loss` puntual -- el ruido de ensayos aleatorios sigue siendo el
cuello de botella dominante, no la arquitectura (el chequeo de cordura de la
entrada 5 ya prueba que puede polarizar fuerte con un solo ensayo fijo).
Ninguna de las dos corridas cruza el umbral todavía. Resultados completos
archivados en `data/interim/{pilot_results,train_log}_{1500ep_16trials,
2500ep_32trials}.{json,txt}`.

**Siguiente paso:** pendiente de decidir con Martín cuál de las vías
priorizar (más `trials_per_step` todavía, más épocas, o revisar
`recurrent_gain`/`tau` en `model.py`/`task.py` en vez de seguir escalando
ruido por fuerza bruta) -- cada duplicación de `trials_per_step` duplica el
coste por época, así que vale la pena decidir con criterio antes de lanzar
una tercera corrida.

## 2026-09-16 (10) — Tres hipótesis baratas descartadas: tau/gain, ring_angle, momentum

**Qué se hizo:** antes de seguir escalando `trials_per_step` (caro), se probaron
tres ajustes de bajo coste computacional sobre la config 2500ep/32trials:

1. **Barrido `tau`/`recurrent_gain`** (4 configs, 400 épocas c/u): sin ganador
   claro, todas en el mismo rango de ruido que el baseline
   (`data/interim/tau_gain_sweep.json`).
2. **Corrección de `ring_angle`** (desfase L/R intercalado a 22.5°, en vez de
   180°, basado en Hulse et al. 2021 eLife 2021;10:e66039): **empeoró mucho**
   la polarización (6.0% -> 0.05% en `frac_polarized_gt_0.9`, misma config
   2500ep/32trials). Revertido -- la fuente solo confirma el desfase
   agregado entre hemisferios, no el patrón de intercalado exacto dentro de
   cada uno, y la versión probada probablemente rompió la continuidad
   angular intra-hemisferio. Resultado archivado como referencia en
   `data/interim/{pilot_results,train_log}_2500ep_32trials_ringfix_WORSE.{json,txt}`.
   **Nota de proceso:** el checkpoint (.pt) de la mejor corrida hasta ahora
   (2500ep/32trials, ring_angle original, 6.0%) no se preservó al lanzar las
   corridas siguientes -- solo se archivaron json/log, no el `.pt`. Si hace
   falta ese checkpoint exacto habría que re-entrenar esa config.
3. **Barrido de `beta1` de Adam** (momentum temporal: 0.9 / 0.97 / 0.99, 400
   épocas c/u, gratis en cómputo): tampoco mostró mejora consistente
   (`data/interim/momentum_sweep.json`).

**Lectura:** ninguna de las tres hipótesis baratas explica el techo de
polarización. El único lever que sí mostró señal real hasta ahora sigue
siendo escalar `trials_per_step` (16 -> 32: 2.1% -> 6.0%), con retornos
decrecientes y coste creciente. `train.py` quedó parametrizado para aceptar
`tau`, `recurrent_gain` y `adam_betas` (antes hardcodeados), útil para
retomar cualquiera de estos experimentos sin tocar el código de nuevo.

**Siguiente paso:** pendiente de decidir con Martín -- seguir escalando
`trials_per_step` (64+, caro), probar una idea estructural distinta (p. ej.
curriculum learning: empezar con ensayos más cortos/fáciles), o parar aquí
por hoy y retomar con la cabeza fresca. El umbral de polarización (10%)
sigue sin cruzarse tras ~4 corridas largas y 3 barridos cortos.

## 2026-09-16 (11) — trials_per_step=64 rompe la tendencia; hallazgo metodológico y corte de sesión

**Qué se hizo:** siguiendo la recomendación de escalar cómputo, se lanzó
`trials_per_step=64` (2500 épocas, mismo resto de config que las corridas
anteriores).

**Resultado:** `frac_polarized_gt_0.9` = **0.79%** -- peor que la corrida de
32 ensayos (6.0%), rompiendo la tendencia creciente que se venía observando
(16 -> 2.1%, 32 -> 6.0%, 64 -> 0.79%). `mean_abs_sign` = 0.330, similar a la
corrida de 32 (0.351). Archivado en
`data/interim/{pilot_results,train_log}_2500ep_64trials.{json,txt}`.

**Hallazgo metodológico (importante):** todas las corridas y barridos de hoy
usaron `seed=0` fijo (misma inicialización de `sign_param` siempre) pero
**una sola repetición por configuración** -- los ensayos aleatorios en sí
sí cambian entre configs (la semilla de `generate_trial` depende de
`epoch * trials_per_step + i`), así que cada config termina viendo una
secuencia de ensayos distinta. Con n=1 por configuración, no se puede
separar "efecto real del hiperparámetro" de "esta corrida en particular
sacó una secuencia de ensayos más fácil/difícil por azar". Esto probablemente
explica por qué la tendencia de `trials_per_step` se rompió al pasar de 32 a
64: es muy posible que gran parte de las diferencias observadas hoy entre
configuraciones (tau/gain, ring_angle, momentum, trials_per_step) sea ruido
de comparación, no señal real.

**Decisión:** cortar la sesión de experimentación aquí. Ya se gastó cómputo
considerable (4 corridas largas + 3 barridos cortos) persiguiendo señales
que probablemente están confundidas con este problema metodológico. Seguir
ajustando hiperparámetros sin arreglar esto primero es de bajo valor.

**PRÓXIMOS PASOS (en orden, para la próxima sesión):**
1. Antes de seguir tocando hiperparámetros: fijar un conjunto de ensayos de
   VALIDACIÓN held-out (misma secuencia de ensayos para todas las
   configuraciones que se comparen, generados con semillas fijas e
   independientes del entrenamiento) para poder medir polarización/pérdida
   de forma comparable entre configs, no confundida con qué ensayos les
   tocaron.
2. Idealmente, correr cada configuración candidata con 2-3 semillas de
   inicialización distintas antes de sacar conclusiones sobre qué
   hiperparámetro "funciona mejor" -- un solo run no alcanza dado el nivel
   de ruido visto hoy.
3. Con esa metodología corregida, retomar la pregunta abierta: ¿qué hace
   falta para cruzar el umbral de polarización (`mean_abs_sign > 0.5`,
   `frac_polarized_gt_0.9 > 0.1`)? Candidatos ya explorados sin éxito claro:
   tau/recurrent_gain, ring_angle (revertido, ver entrada 10), momentum de
   Adam, escalar trials_per_step (16/32/64, no monótono). Sin explorar
   todavía: curriculum learning (empezar con ensayos más cortos/fáciles).
4. Pendiente aparte, no bloqueante: seguir sin verificar la tabla real
   glomérulo-cuña de Hulse et al. (2021, Fig. 10) para `ring_angle` -- el
   intento de corrección de hoy (entrada 10) se descartó por falta de
   precisión, no por descartar que el problema exista.

**Archivos que importan para retomar:** este cuaderno (léelo entero),
`src/cx_net/train.py` (ahora acepta `tau`, `recurrent_gain`, `adam_betas`
como parámetros, además de `trials_per_step`/`n_epochs`/`patience`),
`data/interim/*_sweep*.json` y `data/interim/*_ep*trials*.{json,txt}`
(resultados de todas las corridas/barridos de hoy, para no repetir
experimentos ya hechos).

## 2026-09-17 — Metodología corregida: la varianza entre semillas explica la falsa tendencia; hipótesis nueva sobre la causa real

**Qué se hizo:** se corrigió el problema metodológico de la entrada (11)
(`seed` ahora también desplaza la secuencia de ensayos de entrenamiento, no
solo la inicialización -- `train.py`) y se agregó un set de validación
held-out fijo (`task.py: generate_held_out_set`, 30 ensayos, semillas
reservadas y disjuntas de cualquier semilla de entrenamiento) para poder
comparar configuraciones de forma limpia. Con esto, se repitió la
comparación `trials_per_step` 16 vs 32 con 3 semillas cada uno (1000 épocas,
presupuesto reducido respecto a las corridas de 2500 de ayer).

**Resultado (media ± desvío sobre 3 semillas):**

| | `frac_polarized_gt_0.9` | `mean_abs_sign` | `held_out_loss` |
|---|---|---|---|
| tps=16 | 0.72% ± 0.63% | 0.307 ± 0.006 | 0.502 ± 0.006 |
| tps=32 | 2.32% ± 2.05% | 0.334 ± 0.047 | 0.498 ± 0.005 |

La diferencia entre medias es menor que el desvío estándar dentro de cada
grupo (especialmente en tps=32) -- **no hay evidencia estadística sólida de
que `trials_per_step` 16 vs 32 cambie el resultado final.** Confirma la
sospecha de la entrada (11): la tendencia creciente que se vio ayer
(16->32->64: 2.1%->6.0%->0.79%) era mayormente ruido de comparar una sola
corrida por configuración, no un efecto real y monótono de escalar el
batch. Detalle completo en `data/interim/multiseed_sweep_summary.json`.

**Hallazgo más importante (cambia el diagnóstico):** `held_out_loss` es
prácticamente idéntico entre configs (~0.50) y muy por debajo del nivel de
azar (~0.85-1.15 antes de entrenar) -- la red SÍ aprende a resolver la
tarea de integración de rumbo razonablemente bien, de forma consistente.
Pero el signo de las aristas individuales sigue débilmente decidido
(`mean_abs_sign` ~0.31-0.33, lejos del umbral 0.5) **en absolutamente todas
las configuraciones probadas hasta ahora** (tau/gain, momentum, batch size,
~10 corridas/barridos en total entre ayer y hoy).

**Hipótesis nueva (la más plausible con la evidencia acumulada):** la tarea
de integración de rumbo, tal como está planteada, probablemente
**subrestringe el signo de cada sinapsis individual** -- muchas
asignaciones de signo distintas pueden lograr un desempeño agregado similar
en la tarea (población de compás + integración), así que el gradiente no
tiene presión real para comprometerse con un signo "decidido" por arista
más allá de cierto punto, sin importar el hiperparámetro de entrenamiento.
Esto NO es un problema de optimización (que es lo que se estuvo asumiendo
y descartando en las entradas 9-11) sino, potencialmente, de diseño de
tarea: se necesitaría una tarea más exigente/restrictiva (múltiples
condiciones, ruido, quizás múltiples tareas simultáneas, más cercano a lo
que la mosca real resuelve con este circuito) para forzar que el signo de
cada arista importe individualmente.

**Siguiente paso (pendiente de decidir con Martín):** (a) evaluar H1 ahora
mismo con el mejor modelo disponible, documentando explícitamente que la
polarización es débil y el test probablemente está subpotenciado -- un
resultado honesto con esa salvedad sigue siendo válido para la propuesta
original; o (b) invertir en rediseñar la tarea para que exija más
compromiso de signo antes de evaluar H1 (esfuerzo mayor, sesión aparte).

## 2026-09-17 (2) — Evaluación de H1 sobre el mejor modelo del sweep limpio: no soportada, pero subpotenciada

**Qué se hizo:** se corrió `evaluate.py` sobre `tps32_seed0` (la corrida con
mayor `frac_polarized_gt_0.9` del sweep limpio de 6 corridas: 4.1%).

**Resultado:** acuerdo observado 0.482, media del nulo (permutación de
etiqueta de NT) 0.490, desvío del nulo 0.005, **p=0.93** -- H1 no soportada.
El acuerdo observado ni siquiera queda por debajo del nulo de forma
significativa (a ~1.4 desvíos, dentro del rango esperable por azar).
Desglose H2 por tipo celular sin señal clara en ningún tipo (rango
0.42-0.53, todo compatible con azar). Detalle completo en
`data/interim/h1_evaluation_tps32_seed0.json`.

**Salvedad importante (no es un resultado fuerte todavía):** como se
documentó en la entrada anterior, este modelo tiene polarización de signo
débil (`mean_abs_sign` 0.333, lejos del umbral 0.5 usado como referencia de
"señal fiable" desde la entrada 8) -- la mayoría de los signos siguen cerca
de su inicialización aleatoria en vez de haberse comprometido con un valor
según el gradiente. Un test de acuerdo de signo sobre una red que apenas
decidió sus signos es, por diseño, de bajo poder estadístico: no se puede
distinguir todavía "H1 es falsa" de "el modelo no llegó a un punto donde
evaluar H1 sea informativo". Este resultado queda registrado como
preliminar/no concluyente, no como evidencia sólida contra H1.

**Siguiente paso:** la vía más prometedora identificada hoy es rediseñar la
tarea de entrenamiento para que exija más compromiso de signo por arista
(ver hipótesis de la entrada anterior: la tarea actual probablemente
subrestringe el signo individual) -- pendiente para una sesión futura con
tiempo dedicado a ese diseño, no un ajuste rápido de hiperparámetros.

## 2026-09-17 (3) — Rediseño de tarea (hold_prob): primera mejora real y reproducible

**Qué se hizo:** implementado `hold_prob` en `generate_trial` (`task.py`):
intercala tramos de velocidad angular cero (quietud forzada) en la traza de
entrenamiento -- durante esos tramos la red debe sostener el bump de rumbo
solo con su propia dinámica recurrente, sin ayuda de entrada externa
(persistent activity de ring attractor), lo que depende más directamente de
inhibición lateral tipo Delta7 que la integración pura. `hold_prob=0`
reproduce exactamente el comportamiento original. Chequeo de cordura previo
(`hold_prob=0.3`, un ensayo) convergió limpio a loss 0.0057, confirmando
que la tarea más difícil sigue siendo resoluble antes de invertir en
entrenamiento estocástico completo.

Se corrió el mismo protocolo limpio de la entrada anterior (3 semillas,
`trials_per_step=16`, 1000 épocas) con `hold_prob=0.3`, comparable
directamente contra el baseline (`hold_prob=0`) ya medido.

**Resultado:**

| | `mean_abs_sign` | `frac_polarized_gt_0.9` | `held_out_loss` |
|---|---|---|---|
| baseline (hold_prob=0) | 0.307 ± 0.006 | 0.72% ± 0.63% | 0.502 ± 0.006 |
| hold_prob=0.3 | **0.349 ± 0.010** | 1.28% ± 0.55% | 0.603 ± 0.009 |

`mean_abs_sign` mejora de forma clara y **sin superposición** entre los
valores individuales de los dos grupos (0.339-0.358 vs 0.300-0.312) --
primera mejora de la sesión (entre las ~12 configuraciones probadas ayer y
hoy) que es estadísticamente convincente, no ruido de comparación.
`frac_polarized_gt_0.9` mejora también pero con más superposición, señal
más débil. `held_out_loss` sube (tarea intrínsecamente más difícil,
esperable, no comparable 1:1 con el baseline). Detalle completo en
`data/interim/holdprob_sweep_summary.json`.

**Lectura:** confirma la hipótesis de la entrada anterior -- exigir que la
red sostenga el bump sin entrada externa fuerza más compromiso de signo que
la integración pura. Sigue lejos del umbral (0.35 vs objetivo 0.5), pero es
la primera dirección con evidencia sólida de funcionar.

**Siguiente paso:** probar `hold_prob` más alto (0.5) con el mismo
protocolo, para ver si el efecto sigue una tendencia dosis-respuesta o si
satura pronto.

## 2026-09-17 (4) — hold_prob=0.5 no mejora más: la tendencia no es monótona

**Qué se hizo:** mismo protocolo limpio (3 semillas, tps=16, 1000 épocas)
con `hold_prob=0.5`, para ver si el efecto positivo de `hold_prob=0.3`
seguía una tendencia dosis-respuesta.

**Resultado:**

| | `mean_abs_sign` | `frac_polarized_gt_0.9` |
|---|---|---|
| hold_prob=0.3 | 0.349 ± 0.010 | 1.28% ± 0.55% |
| hold_prob=0.5 | 0.351 ± 0.076 | 0.88% ± 1.21% |

Media prácticamente igual, pero el desvío se disparó (0.076 vs 0.010;
valores individuales 0.430/0.344/0.278) -- no hay ganancia sistemática al
subir más `hold_prob`, solo más varianza entre semillas. Detalle en
`data/interim/holdprob05_sweep_summary.json`.

**Lectura:** `hold_prob=0.3` parece un punto razonable dentro de lo
explorado; seguir subiendo la dificultad de la tarea no es la vía de
mejora obvia. La ganancia real está en haber pasado de "sin tramos de
quietud" a "con tramos de quietud", no en cuánta quietud exactamente.

**Siguiente paso:** en vez de seguir buscando variantes de `hold_prob`,
usar la configuración ya validada (`hold_prob=0.3`, tps=16) con un
presupuesto de entrenamiento mayor (2500 épocas, como las corridas
exploratorias de ayer) para obtener el mejor modelo posible bajo este
diseño de tarea y evaluar H1 sobre él.

## 2026-09-17 (5) — Escalado de hold_prob=0.3 a 2500 épocas + evaluación H1: mejora real pero insuficiente

**Qué se hizo:** se escaló la config validada (`hold_prob=0.3`, `tps=16`) de
1000 a 2500 épocas, mismas 3 semillas, y se evaluó H1 sobre la mejor
(seed0).

**Resultado entrenamiento:** `mean_abs_sign` 0.364 ± 0.029 (mejor semilla:
0.398), `frac_polarized_gt_0.9` 2.05% ± 1.21% (mejor: 3.14%) -- mejora
modesta sobre las 1000 épocas (0.349 ± 0.010) pero sigue lejos del umbral
(0.5 / 10%). Detalle en `data/interim/holdprob03_long_summary.json`.

**Resultado H1** (sobre `hold03_long_seed0`): acuerdo observado 0.483,
media del nulo 0.495, **p=0.99** -- no soportada, mismo patrón que la
evaluación anterior (entrada 2026-09-17 (2)): sin señal en ningún tipo
celular del desglose H2. Detalle en
`data/interim/h1_evaluation_hold03_long_seed0.json`.

**Balance del rediseño de tarea (hold_prob):** primera mejora real y
reproducible de toda la sesión de dos días (0.307 -> 0.364 en
`mean_abs_sign`, confirmada con metodología limpia de múltiples semillas),
pero la magnitud del efecto es insuficiente por sí sola para cruzar el
umbral de polarización necesario para que un test de H1 sea informativo.
Se necesitaría un salto de diseño más grande (no una variante más de
`hold_prob`, ya se probó que subir la dificultad no sigue una tendencia
dosis-respuesta limpia -- ver entrada (4)) para cerrar la brecha.

**Siguiente paso (decisión de alcance, no ajuste técnico menor):**
considerar un rediseño de tarea más sustancial en una sesión dedicada --
por ejemplo, múltiples tareas/condiciones simultáneas (no solo mantener el
bump, sino también responder a perturbaciones, señales visuales, o
distintos regímenes de velocidad) que se acerque más a la variedad de
demandas que el circuito real resuelve. Alternativa: aceptar la limitación
actual y reportarla como parte de la discusión metodológica del preprint
(un hallazgo de que la tarea de heading-integration por sí sola no alcanza
para restringir el signo synáptico individual es, en sí mismo, un
resultado científico válido y publicable).

## 2026-09-17 (6) — Corte de sesión: estado y próximos pasos

**Por qué se corta aquí:** fin de la sesión de dos días de debugging
metodológico y rediseño de tarea. Todo el trabajo quedó commiteado
localmente (8 commits, sin `git push` -- pendiente de decidir si publicar
el repo o mantenerlo privado por ahora).

**ESTADO EXACTO al cortar:**
- ✅ Metodología de comparación corregida y validada: `seed` ahora varía
  también la secuencia de ensayos de entrenamiento (no solo la
  inicialización), y hay un set de validación held-out fijo
  (`task.py: generate_held_out_set`, semillas ≥ 900.000.000, disjuntas de
  cualquier semilla de entrenamiento). `train.py` acepta `run_label` --
  cada corrida guarda `model_<label>.pt` / `results_<label>.json` sin
  pisar corridas anteriores.
- ✅ Se descartaron con evidencia sólida (multi-semilla, no una sola
  corrida) como causa del techo de polarización de signo: `tau`,
  `recurrent_gain`, momentum de Adam, y escalar `trials_per_step` (16 vs
  32 vs 64 -- ninguna diferencia estadísticamente convincente).
- ✅ **Único hallazgo positivo real de las dos sesiones:** `hold_prob`
  (tramos de quietud forzada en la traza de entrenamiento, `task.py`) sube
  `mean_abs_sign` de 0.307±0.006 a 0.349-0.364 de forma reproducible
  (sin superposición entre semillas). No sigue tendencia dosis-respuesta
  limpia (`hold_prob=0.5` no mejora sobre `0.3`, solo agrega varianza).
- ❌ **H1 sigue sin poder evaluarse de forma concluyente.** Se evaluó dos
  veces (sobre `tps32_seed0` y sobre `hold03_long_seed0`, el mejor modelo
  disponible) -- ambas veces p>0.9, sin señal en ningún tipo celular. Pero
  la polarización de signo en ambos modelos sigue por debajo del umbral de
  referencia (`mean_abs_sign>0.5`, `frac_polarized_gt_0.9>10%`), así que
  esto es un resultado preliminar/subpotenciado, NO evidencia sólida de que
  H1 sea falsa.
- 📝 Discusión aparte (no bloqueante para la investigación): estrategia de
  difusión en LinkedIn/CV. Conclusión de esa charla: no esperar a cerrar H1
  para publicar -- el ángulo con más alcance para audiencia general es la
  pregunta científica en sí ("¿puede una IA adivinar la química de un
  cerebro solo mirando el cableado?"), apalancando que el connectome
  completo de MaleCNS se publicó hace apenas dos semanas (2026-09-03). El
  ángulo metodológico (bug de semillas, validación multi-semilla) queda como
  posible contenido secundario para audiencia técnica, no como gancho
  principal. Pendiente: redactar el borrador si Martín quiere retomarlo.

**PRÓXIMOS PASOS (en orden, para retomar):**
1. Decisión de alcance pendiente (la más importante): ¿invertir en un
   rediseño de tarea más sustancial (múltiples condiciones/comportamientos
   simultáneos, no solo `hold_prob`) para intentar cruzar el umbral de
   polarización, o aceptar la limitación actual y reportarla como hallazgo
   metodológico en la discusión del preprint? Esto es una decisión de
   alcance de proyecto, no un ajuste técnico -- requiere tiempo dedicado,
   no un arranque rápido de sesión.
2. Si se sigue con el rediseño de tarea: partir de `hold_prob=0.3` como
   base validada (no repetir `tau`/`gain`/momentum, ya descartados con
   evidencia sólida). Candidatos no probados todavía: múltiples tareas
   simultáneas, perturbaciones sensoriales, entrenar sobre un rango más
   amplio de condiciones de una sola vez.
3. Pendiente aparte, técnico, de menor prioridad: vectorizar el `for` de
   `trials_per_step` en `train.py` (ensayo por ensayo -> batch) para
   acelerar el entrenamiento en CPU -- no se probó GPU porque la forma del
   cómputo (RNN secuencial de tensores chicos) no encaja bien con GPU, y el
   hardware disponible (AMD en Windows) tiene mal soporte para PyTorch de
   todos modos.
4. Pendiente aparte, no bloqueante desde la entrada (4) del 2026-09-16:
   seguir sin verificar la tabla real glomérulo-cuña de Hulse et al. (2021,
   Fig. 10) para `ring_angle` -- el intento de corrección del 2026-09-16
   (entrada 10) se descartó por falta de precisión, no por descartar que
   el problema exista.

**Archivos que importan para retomar:** este cuaderno (léelo entero de
arriba a abajo), `src/cx_net/*.py`, `data/interim/holdprob03_long_summary.json`
(mejor resultado de entrenamiento hasta ahora), `data/interim/h1_evaluation_
hold03_long_seed0.json` (última evaluación de H1). Los `.pt`/`.json` con
prefijos `tps16_`/`tps32_`/`hold03_`/`hold05_`/`_seed{0,1,2}` en
`data/interim/` son resultados de sweeps ya interpretados y registrados --
no hace falta re-correrlos, solo consultarlos si hace falta el detalle
crudo.

## 2026-09-17 (7) — Segunda condición simultánea (`perturb_amp`): primer resultado de H1 estadísticamente significativo

**Qué se hizo:** siguiendo la decisión de alcance pendiente de la entrada (6)
(invertir en un rediseño de tarea más sustancial en vez de aceptar la
limitación actual), se implementó `perturb_amp` en `generate_trial`
(`task.py`): ruido gaussiano iid de alta frecuencia y media cero añadido al
canal de entrada externa (PEN_a/PEN_b) que NO cuenta para el `heading`
objetivo -- la red debe integrar la señal real Y rechazar el ruido al mismo
tiempo, lo que depende del filtrado temporal (`tau`) e inhibición lateral,
un mecanismo distinto al de `hold_prob` (memoria persistente sin entrada).
Se aplica en TODOS los pasos, incluidos los tramos de quietud de
`hold_prob`, así que combinar ambos es la condición más exigente probada en
el proyecto hasta ahora.

**Incidencia de proceso (importante, ver memoria del proyecto):** el primer
intento de chequeo de cordura dio NaN inmediato en el primer paso de
gradiente, en TODAS las configuraciones probadas incluido el baseline
original (`hold_prob=0`, ya validado muchas veces antes). Diagnóstico:
se estaba invocando el `python` global del sistema (torch 2.4.1, instalación
no relacionada) en vez de `.venv/Scripts/python.exe` (torch 2.14.0) -- el
NaN era una interacción de ese torch ajeno con el punto exacto-cero de
`atan2` en `decode_heading` (el estado de la población brújula en t=0 es
matemáticamente cero siempre, por construcción), no un bug real del código.
Con el `.venv` correcto, el baseline original entrena con normalidad. Queda
anotado en memoria para no repetir la confusión: invocar siempre
`.venv/Scripts/python.exe` explícitamente.

**Chequeo de cordura (sobreajuste de un ensayo, `hold_prob=0.3` fijo,
barriendo `perturb_amp`):** dosis-respuesta clara y monotónica en
`mean_abs_sign`: 0.35 (amp=0.02) -> 0.42 (0.04) -> 0.47 (0.08) -> 0.66
(0.12) -> 0.58 (0.16) -> 0.61 (0.2), todas por encima del baseline de
`hold_prob=0.3` solo (0.50) salvo las amplitudes más bajas. Se eligió
`perturb_amp=0.12` para el barrido completo (punto más alto y estable antes
de que 0.16/0.2 empezaran a mostrar entrenamiento errático -- plateaus
largos seguidos de caídas súbidas, señal de LR demasiado alto para esa
dificultad).

**Barrido completo (`hold_prob=0.3` + `perturb_amp=0.12`, protocolo
idéntico al de la entrada anterior: 3 semillas, `tps=16`, 1000 épocas,
`lr=0.05`):**

| semilla | `mean_abs_sign` | `frac_polarized_gt_0.9` | `held_out_loss` |
|---|---|---|---|
| 0 | 0.327 | 1.5% | 0.864 |
| 1 | 0.424 | 3.7% | 0.904 |
| 2 | **0.645** | **12.8%** | 1.062 |
| media ± dt | 0.465 ± 0.163 | 6.0% ± 6.0% | 0.943 ± 0.105 |

La media sube claramente sobre cualquier configuración anterior (0.307
baseline, 0.364 mejor `hold_prob` solo), pero con una varianza mucho mayor
que cualquier sweep previo (dt 0.163 vs ~0.01-0.03 en sweeps anteriores) --
la condición combinada parece tener un comportamiento más bimodal
(polariza fuerte o no) que un efecto uniforme entre semillas.
`held_out_loss` sube (tarea mucho más difícil, no comparable 1:1 con
configs anteriores, esperable). Detalle completo en
`data/interim/holdperturb_sweep_summary.json`.

**Hito del proyecto: la semilla 2 es el PRIMER modelo en cruzar el umbral
de referencia** (`mean_abs_sign>0.5`, `frac_polarized_gt_0.9>10%`) en las
~15 configuraciones probadas desde la entrada (8) del 2026-09-16.

**Evaluación de H1 sobre las 3 semillas:**

| semilla | `observed_agreement` | `null_mean` | p | ¿pasa el umbral de polarización? |
|---|---|---|---|---|
| 0 | 0.483 | 0.494 | 0.988 | No |
| 1 | 0.477 | 0.489 | 0.990 | No |
| 2 | **0.547** | 0.519 | **0.0005** | **Sí** |

Las semillas 0 y 1 (sin cruzar el umbral) dan el mismo patrón de ruido puro
visto en todas las evaluaciones anteriores del proyecto -- consistente con
la hipótesis de que el test de H1 no es informativo sin polarización
suficiente. La semilla 2 (la única que cruza el umbral) da el **primer
resultado de H1 estadísticamente significativo de todo el proyecto**:
acuerdo observado 54.7% vs. 51.9% esperado por azar, p=0.0005. Desglose H2:
señal más fuerte en PEN_b(PEN2) (65.8%) y PEN_a(PEN1) (60.3%), más débil en
Delta7 (50.6%, sin señal) y EPGt (46.1%, por debajo del azar pero con solo
152 aristas). Detalle completo en
`data/interim/h1_evaluation_holdperturb_seed{0,1,2}.json`.

**Por qué este resultado NO se acepta todavía como evidencia sólida de
H1:** es una sola semilla de tres, y esa semilla fue también la que más se
alejó del resto en polarización (0.645 vs. 0.327/0.424) -- exactamente el
patrón de varianza alta entre semillas que el proyecto ya identificó como
razón para no confiar en corridas únicas (entrada 11 del 2026-09-16). El
tamaño del efecto además es modesto (2.8 puntos porcentuales sobre el
azar), aunque estadísticamente muy significativo gracias al gran número de
aristas (9.160). Que las dos semillas SIN polarización suficiente den
ruido puro (p~0.99) mientras que la única con polarización suficiente dé
señal fuerte es alentador -- coincide con la teoría del proyecto de que el
umbral de polarización es el filtro correcto -- pero con n=1 "réplica
exitosa" no se puede distinguir todavía "efecto real reproducible que
necesita la semilla correcta" de "una coincidencia de una corrida entre
tres".

**Siguiente paso (decisión pendiente, no bloqueante):** correr semillas
adicionales (p. ej. 3-7) en la misma configuración (`hold_prob=0.3`,
`perturb_amp=0.12`) para ver qué fracción cruza el umbral de polarización
de forma consistente y si el signo del efecto de H1 se repite en las que sí
cruzan -- antes de escribir cualquier conclusión sobre H1 en el preprint.

## 2026-09-18 — Réplica con 5 semillas más: el hallazgo de H1 de la semilla 2 no se sostiene

**Qué se hizo:** se corrieron 5 semillas adicionales (3-7) con la misma
configuración de la entrada anterior (`hold_prob=0.3`, `perturb_amp=0.12`,
protocolo idéntico) para ver si la única semilla que había cruzado el
umbral de polarización (semilla 2, con el primer resultado de H1
significativo del proyecto) representaba un efecto real o una corrida
atípica.

**Resultado (8 semillas en total, 0-7):**

| | `mean_abs_sign` | `frac_polarized_gt_0.9` |
|---|---|---|
| valores individuales | 0.327, 0.424, **0.645**, 0.343, 0.370, 0.462, 0.361, 0.388 |
| media ± dt | 0.415 ± 0.103 | 4.4% ± 4.4% |
| **semillas que cruzan el umbral** | **1 de 8** (solo la semilla 2) | |

Comparado contra `hold_prob=0.3` solo (0.349 ± 0.010, sin solapamiento
individual entre semillas -- el criterio que el proyecto ya usó para
aceptar un efecto como real, ver entrada 2026-09-17 (3)): aquí SÍ hay
solapamiento amplio -- 5 de las 8 semillas nuevas (0.327, 0.343, 0.370,
0.361, 0.388) caen dentro o por debajo del rango que ya daba `hold_prob`
solo. Solo 3 de 8 (0.424, 0.462, 0.645) superan claramente ese rango, y
únicamente la más extrema (0.645) llega a cruzar el umbral de referencia.
Detalle completo en `data/interim/holdperturb_sweep_summary_8seeds.json`.

**Conclusión (aplicando el mismo estándar de rigor que el proyecto ya usó
para aceptar el hallazgo de `hold_prob`):** `perturb_amp=0.12` combinado
con `hold_prob=0.3` NO pasa la prueba de "sin solapamiento entre grupos" --
a diferencia de `hold_prob` solo (que sí la pasó y se aceptó como hallazgo
real), aquí el efecto medio está confundido con un aumento grande de la
varianza entre semillas (dt 0.103 vs. 0.01 de `hold_prob` solo). En
consecuencia, **el resultado de H1 significativo de la semilla 2 (entrada
anterior, p=0.0005) se retracta como evidencia de H1**: con 8 semillas
probadas y solo 1 cruzando el umbral de forma aislada, es indistinguible
de una corrida que polarizó fuerte por azar de optimización, no de un
efecto reproducible de la condición de tarea. No se promueve a resultado
del preprint.

**Balance acumulado de los dos días de rediseño de tarea (`hold_prob`,
`perturb_amp`, ~20 configuraciones distintas probadas entre las dos
sesiones):** un solo hallazgo pasa el estándar de rigor del proyecto
(`hold_prob` sube `mean_abs_sign` de 0.307 a 0.349-0.364 de forma limpia,
sin solapamiento) y por sí solo es insuficiente para cruzar el umbral de
polarización necesario para un test de H1 informativo. Ninguna otra
variante probada (tau/gain, momentum, `trials_per_step`, `perturb_amp`,
combinaciones) mejora sobre eso de forma reproducible -- en el mejor de los
casos (`perturb_amp`) solo aumenta la varianza, sin desplazar la media de
forma confiable.

**Siguiente paso (decisión de alcance, la misma que quedó pendiente en la
entrada 2026-09-17 (6), ahora con más evidencia para decidirla):** dado
que ~20 configuraciones a lo largo de dos sesiones no lograron un efecto
de diseño de tarea que cruce el umbral de forma reproducible, la
recomendación es cerrar la fase de rediseño de tarea aquí y reportar la
limitación como hallazgo metodológico legítimo -- pendiente de decidir con
Martín si evaluar H1 con el mejor modelo consistente disponible
(`hold03_long_seed0` o el mejor de `hold_prob` solo) documentando
explícitamente la subpotencia, o cerrar la fase experimental sin una
evaluación final de H1 y pasar directamente a redactar la discusión
metodológica del preprint.

## 2026-09-18 (2) — Corte de sesión: estado y próximos pasos

**Por qué se corta aquí:** cierre de la sesión de dos días de rediseño de
tarea (`hold_prob`, `perturb_amp`). Todo el código y el cuaderno quedan
guardados y commiteados localmente antes de cortar (ver commits de hoy).

**ESTADO EXACTO al cortar:**
- ✅ `perturb_amp` implementado y documentado en `task.py`/`train.py`/
  `sanity_check.py` (ruido de alta frecuencia en el canal de entrada,
  excluido del `heading` objetivo -- ver entrada 2026-09-17 (7) para el
  razonamiento completo).
- ✅ **Hallazgo positivo confirmado y aceptado (el único que pasa el
  estándar de rigor del proyecto):** `hold_prob=0.3` solo sube
  `mean_abs_sign` de 0.307±0.006 a 0.349-0.364, sin solapamiento entre
  semillas -- sigue siendo la mejor configuración validada del proyecto.
- ❌ **`perturb_amp=0.12` (combinado con `hold_prob=0.3`) NO se acepta
  como hallazgo real** tras la réplica con 8 semillas totales (0-7): solo
  1 de 8 cruza el umbral de polarización, y hay solapamiento amplio con el
  rango de `hold_prob` solo -- el resultado de H1 significativo de la
  semilla 2 (p=0.0005) queda retractado, indistinguible de una corrida que
  polarizó fuerte por azar de optimización. Detalle completo en
  `data/interim/holdperturb_sweep_summary_8seeds.json` y
  `data/interim/h1_evaluation_holdperturb_seed{0-2}.json`.
- ❌ **H1 sigue sin poder evaluarse de forma concluyente** con ninguna
  configuración probada hasta ahora (3 evaluaciones en total a lo largo del
  proyecto: `tps32_seed0`, `hold03_long_seed0`, `holdperturb_seed2` --
  todas p>0.9 o retractadas, ninguna sobrevive como evidencia sólida).
- 🐛 **Gotcha de entorno encontrado y corregido en memoria** (no en el
  código, es un problema de invocación): un `python`/`pip` sin calificar en
  este entorno puede resolver a un torch ajeno (2.4.1) en vez del `.venv`
  del proyecto (2.14.0), produciendo un NaN de gradiente reproducible que
  parece un bug real pero no lo es. Invocar siempre
  `.venv/Scripts/python.exe` explícitamente -- ver memoria de proyecto
  (`project_cx_net.md`) y entrada 2026-09-17 (7).
- 📌 **Decisión de alcance pendiente, la más importante para la próxima
  sesión:** con ~20 configuraciones de rediseño de tarea probadas en dos
  sesiones y solo un hallazgo modesto que no alcanza el umbral, la
  recomendación registrada es cerrar la fase de rediseño de tarea y
  reportar la limitación como hallazgo metodológico legítimo del preprint,
  en vez de seguir buscando una tercera variante de tarea. Falta decidir
  con Martín: (a) evaluar H1 una última vez con el mejor modelo consistente
  disponible (`hold03_long_seed0`, o repetir esa config con más semillas
  para elegir el mejor de forma limpia) documentando la subpotencia
  explícitamente, o (b) cerrar la fase experimental sin una evaluación
  final de H1 y pasar directamente a redactar la discusión metodológica del
  preprint.

**PRÓXIMOS PASOS (en orden, para retomar):**
1. Decidir con Martín la decisión de alcance de arriba (evaluar H1 una
   última vez vs. cerrar sin evaluación final) -- no es un ajuste técnico,
   es una decisión sobre qué va en el preprint.
2. Si se decide evaluar H1 una última vez: usar `hold_prob=0.3` (sin
   `perturb_amp`, ya descartado) como base, posiblemente con más semillas
   que las 3 ya corridas (`hold03_seed{0,1,2}` en
   `data/interim/holdprob_sweep_summary.json`) para elegir el modelo con
   mejor polarización de forma honesta (no cherry-picking post-hoc como
   pasó con la semilla 2 de `perturb_amp`).
3. Si se decide cerrar sin más experimentos: redactar la sección de
   discusión metodológica del preprint documentando que la tarea de
   heading-integration, incluso reforzada con memoria-sin-entrada y
   filtrado de ruido, no restringe lo suficiente el signo sináptico
   individual para un test de H1 informativo -- con las tablas de este
   cuaderno como evidencia (20 configuraciones, un solo efecto reproducible
   y aun así insuficiente).
4. Pendiente aparte, no bloqueante, arrastrado desde el 2026-09-16 (4):
   validar `ring_angle` contra la tabla real glomérulo-cuña de Hulse et al.
   (2021, Fig. 10) antes de confiar en resultados cuantitativos de
   decodificación -- sigue sin abordarse.
5. Pendiente aparte, técnico, de baja prioridad (arrastrado desde
   2026-09-17 (6)): vectorizar el `for` de `trials_per_step` en `train.py`
   para acelerar el entrenamiento en CPU.

**Archivos que importan para retomar:** este cuaderno (léelo entero de
arriba a abajo), memoria de proyecto `project_cx_net.md` (nota sobre el
gotcha de `.venv`), `src/cx_net/task.py` (`perturb_amp`),
`data/interim/holdperturb_sweep_summary_8seeds.json` (resultado que se
retracta, para no repetir el barrido), `data/interim/holdprob_sweep_summary.json`
(el hallazgo que sí se mantiene en pie).

## 2026-09-18 (3) — `sign_reg`: regularización directa del signo, polariza con éxito pero H1 sigue sin evidencia

**Qué se hizo:** en vez de seguir buscando una variante de TAREA que fuerce
polarización de forma emergente (agotado tras `hold_prob`/`perturb_amp`,
ver entradas anteriores), se probó una palanca distinta: una penalización
añadida directamente al gradiente de `sign_param`, independiente de los
ensayos y de la dinámica de la red --
`model.sign_confidence_penalty() = (1 - tanh(sign_param)^2).mean()`, el
término que hace de la derivada de `tanh` -- mínimo cuando cada arista está
polarizada (`|tanh(sign_param)| -> 1`), máximo cuando está indecisa (cerca
de 0). No favorece qué signo tomar, solo penaliza la indecisión, así que la
dirección la sigue decidiendo el gradiente de tarea. Implementado en
`model.py` (`sign_confidence_penalty`), `train.py`/`sanity_check.py`
(`sign_reg`, se suma al gradiente de tarea antes del clipping).

**Chequeo de dosis-respuesta (sobreajuste de un ensayo, `hold_prob=0.3`
fijo, barriendo `sign_reg`):** efecto muy limpio y monotónico, y a
diferencia de `perturb_amp`, casi sin coste en la pérdida de entrenamiento:

| `sign_reg` | `mean_abs_sign` | `frac_polarized_gt_0.9` | `loss_last` |
|---|---|---|---|
| 0 (baseline) | 0.505 | 2.3% | 0.0057 |
| 0.001 | 0.576 | 9.2% | 0.0057 |
| 0.005 | 0.688 | 29.3% | 0.0058 |
| 0.01 | 0.758 | 44.2% | 0.0059 |
| 0.05 | 0.878 | 74.7% | 0.0064 |
| 0.1 | 0.908 | 80.4% | 0.0076 |

Se eligió `sign_reg=0.05` para el barrido completo: polarización muy alta
con la pérdida de overfit de un ensayo apenas por encima del baseline.

**Barrido completo (`hold_prob=0.3` + `sign_reg=0.05`, protocolo idéntico:
`tps=16`, 1000 épocas, `lr=0.05`, 8 semillas 0-7 corridas de una vez dado
lo prometedor del chequeo previo):**

| semilla | `mean_abs_sign` | `frac_polarized_gt_0.9` | `held_out_loss` | `h1_p` |
|---|---|---|---|---|
| 0 | 0.794 | 57.8% | 0.592 | 0.618 |
| 1 | 0.718 | 37.0% | 0.723 | **0.0055** |
| 2 | 0.606 | 26.8% | 0.603 | 0.294 |
| 3 | 0.439 | 8.8% | 0.582 | 0.152 |
| 4 | 0.593 | 22.4% | 0.715 | 0.686 |
| 5 | 0.739 | 52.0% | 0.639 | 0.999 |
| 6 | 0.838 | 69.1% | 0.596 | 0.913 |
| 7 | 0.769 | 51.7% | 0.599 | 0.742 |
| media ± dt | 0.687 ± 0.123 | 40.7% ± 19.1% | 0.631 ± 0.053 | -- |

Detalle completo en `data/interim/signreg05_sweep_summary_8seeds.json` y
`data/interim/h1_evaluation_signreg05_seed{0-7}.json`.

**Hallazgo 1 (aceptado, el mecanismo de polarización más fuerte y limpio
del proyecto):** `mean_abs_sign` NO se solapa ni una vez con el baseline de
`hold_prob=0.3` solo (0.339-0.358) -- incluso la semilla más débil de
`sign_reg` (0.439, semilla 3) casi duplica la semilla más alta del baseline.
7 de 8 semillas cruzan el umbral de polarización de referencia
(`frac_polarized_gt_0.9>10%`), la única excepción (semilla 3, 8.8%) queda
justo por debajo. A diferencia de `perturb_amp`, este efecto pasa el
estándar de "sin solapamiento" del proyecto de forma consistente en las 8
semillas, no en 1 de 8. El coste en `held_out_loss` es real pero moderado y
desigual entre semillas: la media sube de 0.603±0.009 (baseline) a
0.631±0.053, con dos semillas (1 y 4) claramente peores (~0.72) y las otras
seis indistinguibles del baseline -- en la mayoría de semillas, mucha más
polarización no cuesta desempeño de tarea.

**Hallazgo 2 (la parte que no se sostiene): con polarización ya sobrada en
7 de 8 semillas, H1 sigue sin evidencia reproducible.** Solo 1 de 8
semillas (la 1) da un resultado significativo (p=0.0055) -- exactamente el
mismo patrón "1 de N" que se retractó para `perturb_amp` en la entrada
anterior, y perfectamente compatible con el ruido esperado de correr 8
tests de permutación independientes a alfa=0.05 (falso positivo esperado
~0.4 de 8 por puro azar). El resto (7 de 8) da p>0.15, la mayoría p>0.6.
No hay relación aparente entre `held_out_loss` alto y significancia de H1:
la semilla 4 tiene un `held_out_loss` similar al de la semilla 1 (0.715 vs.
0.723) pero p=0.686, así que el resultado de la semilla 1 no se explica por
"peor ajuste de tarea -> más parecido al azar biológico por casualidad" de
forma sistemática -- parece simplemente el resultado significativo aislado
que cualquier barrido de 8 semillas produce por azar.

**Por qué este resultado es más concluyente que los anteriores para la
pregunta de alcance:** todos los intentos previos (`hold_prob`,
`perturb_amp`) dejaban abierta la duda de si la falta de señal de H1 se
debía a polarización insuficiente. Con `sign_reg=0.05`, la polarización ya
no es el cuello de botella (7/8 semillas muy por encima del umbral,
`mean_abs_sign` medio 0.687 frente a 0.349 del mejor hallazgo anterior) y
el patrón de H1 sigue siendo indistinguible de ruido puro. Esto es
evidencia bastante más fuerte que antes de que el problema no es "no hemos
polarizado lo suficiente" sino que el signo aprendido -- polarizado o no --
simplemente no coincide con el neurotransmisor real más de lo esperado por
azar en esta arquitectura/tarea.

**Siguiente paso:** decisión de alcance del cuaderno (ver entrada
2026-09-18 (2)) -- con esta prueba adicional (regularización directa,
mecanismo distinto a los de tarea, sin encontrar señal pese a resolver el
problema de polarización) el caso para cerrar la fase experimental y
reportar la limitación metodológica en el preprint queda más sólido, no
menos.

## 2026-09-18 (4) — Decisión de alcance resuelta: se cierra la fase experimental de rediseño de tarea/regularización

**Decisión:** con `sign_reg` resolviendo el problema de polarización de
forma limpia en 7/8 semillas y H1 seguir sin evidencia reproducible (ver
entrada anterior), se cierra aquí la fase de búsqueda de una condición de
entrenamiento que haga a H1 evaluable, y se pasa a redactar la discusión
metodológica del preprint documentando el resultado negativo. No se
evaluará H1 "una última vez" con más configuraciones -- ya se probaron dos
mecanismos ortogonales (rediseño de tarea: `hold_prob`/`perturb_amp;
regularización directa: `sign_reg`) y ambos, aun cuando uno de ellos
resuelve la polarización por completo, dan el mismo patrón de ausencia de
señal.

**Resumen de la fase completa (tres sesiones, 2026-09-16 a 2026-09-18,
~30 configuraciones distintas probadas):**

| mecanismo | mejor `mean_abs_sign` | ¿pasa "sin solapamiento"? | ¿H1 reproducible? |
|---|---|---|---|
| tarea original (`hold_prob=0`) | 0.307±0.006 | -- (baseline) | No (p~0.99, subpotenciado) |
| `hold_prob=0.3` | 0.349±0.010 | Sí, vs. baseline | No evaluado a fondo (bajo el umbral) |
| `hold_prob=0.3`+`perturb_amp=0.12` | 0.465±0.163 (bimodal) | No, solapa con `hold_prob` solo | No -- 1/8 semillas, retractado |
| `hold_prob=0.3`+`sign_reg=0.05` | 0.687±0.123 | Sí, vs. todo lo anterior | No -- 1/8 semillas, mismo patrón |

**Qué queda establecido para el preprint:** la topología real del
subcircuito CX, combinada con una tarea de integración de rumbo (incluso
reforzada con memoria-sin-entrada y filtrado de ruido) y/o con
regularización directa del signo, permite entrenar redes que resuelven la
tarea y opcionalmente polarizan con fuerza -- pero el signo que aprenden,
esté polarizado o no, no coincide con el neurotransmisor real anotado más
de lo esperable por azar (test de permutación, 2000 permutaciones,
n_aristas=9160). La ausencia de señal no puede atribuirse ya a potencia
estadística insuficiente por falta de polarización -- se descarta
explícitamente con `sign_reg`.

**PRÓXIMOS PASOS (en orden):**
1. ✅ ~~Decisión de alcance~~ -- resuelta arriba.
2. Redactar la sección de discusión metodológica del preprint (siguiente
   tarea, ver `docs/preprint-discussion.md` cuando exista).
3. Pendiente aparte, no bloqueante, arrastrado desde el 2026-09-16 (4):
   validar `ring_angle` contra la tabla real glomérulo-cuña de Hulse et al.
   (2021, Fig. 10) antes de confiar en resultados cuantitativos de
   decodificación.
4. Pendiente aparte, técnico, de baja prioridad (arrastrado desde
   2026-09-17 (6)): vectorizar el `for` de `trials_per_step` en `train.py`
   para acelerar el entrenamiento en CPU (ya no es crítico si no hay más
   barridos grandes planeados, pero facilitaría cualquier verificación
   futura).

## 2026-09-19 — Revisión del borrador de discusión y corte de sesión

**Qué se hizo:** revisión crítica de `docs/preprint-discussion.md`
contrastándolo con este cuaderno, los `h1_evaluation_signreg05_seed*.json` y
`task.py`. Cambios aplicados al borrador (sin commit todavía):

- **Sección 3:** el argumento "1 de 8 es indistinguible del 5% esperado" era
  débil. Con p=0.0055 la probabilidad de que el mínimo de 8 p-valores baje de
  ese valor bajo la nula es ≈4.3%, y 0.0055 < 0.00625 (Bonferroni, 8 tests).
  Lo que sí sostiene el resultado nulo: efecto diminuto en la semilla 1
  (acuerdo 0.519 vs. nula 0.507), semilla 5 con desviación opuesta y mayor
  (z=−3.16, p una cola=0.999, no señalada por el test de una cola), y
  agregado sin señal (z medio −0.22; Stouffer z=−0.62, p=0.73; Fisher
  p=0.26). Se añadió la tabla por semilla con z.
- La retractación de `perturb_amp` (p=0.0005, que también pasaría
  Bonferroni) se hizo por criterio de polarización/replicación, no por
  comparaciones múltiples; el borrador lo dice explícitamente ahora.
- "Convergencia de dos mecanismos ortogonales" sobreestimada: `perturb_amp`
  solo polarizó 1/8, así que la evidencia con potencia es solo `sign_reg`.
- Sección 7 contradecía a la 4 ("no basta" vs. "sin poder para
  confirmar/refutar"); reescrita como "no encontramos evidencia de…".
- "Muy por encima del azar" corregido: held-out ≈0.6 vs. ≈0.85 de azar.
- `ring_angle` SÍ afecta a H1 indirectamente (`task.py` lo usa para el
  objetivo de entrenamiento); el borrador decía lo contrario.
- Aclarado que "8/8" en la tabla es no-solapamiento de `mean_abs_sign`; el
  cruce del umbral es 7/8. (Sin inconsistencia real con el mensaje del
  commit `10b7102`.)
- Limitaciones nuevas: sin control positivo de potencia; grados de libertad
  del investigador (~30 configuraciones, umbrales sin análisis previo);
  solución de tarea solo moderada.

**PRÓXIMOS PASOS (en orden; retomar aquí en la siguiente sesión):**
1. **Control positivo de potencia** (decidido: se hace al empezar la
   próxima sesión). Sembrar signos con acuerdo predefinido (p. ej. 52%,
   55%, 60%) sobre las 9.160 aristas reales, con el mismo procedimiento de
   test de permutación (2000 permutaciones, una cola) y las etiquetas reales
   de neurotransmisor, y medir con qué frecuencia el test lo detecta (curva
   potencia vs. acuerdo sembrado, idealmente con la polarización real de los
   modelos `sign_reg`, `mean_abs_sign`≈0.69). Objetivo: sustituir "potencia
   suficiente" (hoy una inferencia) por una medición. Los resultados y su
   efecto sobre la redacción van a `docs/preprint-discussion.md` (secciones 2
   y 5). Variante más costosa, opcional: entrenar sobre una red con signos
   conocidos.
2. Releer el borrador completo tras incorporar el control positivo y
   comprobar coherencia entre secciones 1, 3, 4, 5 y 7.
3. Commit de `docs/preprint-discussion.md` y de esta entrada.
4. Actualizar `README.md` (dice "Fase de diseño / extracción de datos",
   desactualizado).
5. Pendientes previos, no bloqueantes: validar `ring_angle` contra Hulse et
   al. (2021, Fig. 10) (ahora con más peso, ver arriba); vectorizar el `for`
   de `trials_per_step` en `train.py`.

## 2026-09-19 (2) — Control positivo de potencia del test de H1

**Qué se hizo:** `src/cx_net/power_control.py`. Sobre los signos aprendidos
de los 8 modelos `signreg05`, se siembra el signo de una "verdad" en una
fracción q de neuronas de origen (o de aristas independientes) y se aplica
el mismo test de permutación (2000 perm., una cola, alfa=0.05), 200
repeticiones por modelo y q. Resultados en `data/interim/power_control.json`.

**Corrección de diseño en el camino:** la primera versión usaba solo las
etiquetas reales; con q=0 las 200 repeticiones eran idénticas
(permutaciones fijas) y salió "potencia 0.125" = 1/8 = la semilla 1, no una
tasa de falsos positivos. Se añadió la variante `decoy` (etiquetas barajadas
sorteadas de nuevo en cada repetición): q=0 da 0.049-0.052, calibrado.

**Resultado / número clave (siembra por neurona, `decoy`):** potencia 0.11
(q=0.01, acuerdo 0.503), 0.26 (0.508), 0.43 (0.513), 0.69 (0.523), 0.98
(0.548). ≈50% en acuerdo ≈0.516 (+1.6 pp), ≈80% en ≈0.533 (+3.3 pp). Con
siembra por aristas independientes la potencia es mayor (80% ≈ +1.6 pp).
Acuerdos reales observados: 0.476-0.519, media 0.497. Conclusión: un efecto
≥+3.3 pp se habría detectado; efectos de +1-2 pp no. Variante `real`
similar (hasta ≈0.07 más de potencia en q bajo).

**Límites:** señal sembrada idealizada; mide la potencia del test, no la
capacidad del entrenamiento de recuperar signos (variante "entrenar sobre
red con signos conocidos" no realizada).

**Siguiente paso:** borrador actualizado (secciones 2, 4, 5, 7). Quedan:
releer coherencia completa (1, 3, 4, 5, 7; la sección 1 aún dice que la
potencia "se descartó" como explicación), commit, actualizar `README.md`.

## 2026-09-19 (3) — ¿La tarea selecciona los signos reales? (sin entrenar)

**Qué se hizo:** `src/cx_net/real_sign_task_check.py`. Se fija el signo de
cada arista al del neurotransmisor real de su neurona de origen
(`sign_param=±10`) y se evalúa la pérdida held-out (hold_prob=0.3, mismo set
que `signreg05`). Referencias: los 8 modelos `signreg05` (signo blando y
signo duro) y 500 asignaciones de neurotransmisor barajadas entre neuronas
(mismo recuento). Salida: `data/interim/real_sign_task_check.json`.

**Resultado / número clave:**
- Signos reales: pérdida **1.006**. Barajados (n=500): media 0.983, std
  0.103, rango 0.739-1.216, percentiles 1/5/50/95/99 = 0.76/0.82/0.98/1.15/1.21.
  p(barajado ≤ real) = 0.58: la química real no resuelve la tarea mejor que
  un reparto aleatorio de neurotransmisores.
- Modelos entrenados (signo blando): 0.58-0.72, por debajo del percentil 1
  de los barajados en 8 de 8. Con signo duro sign(sign_param) a |tanh|=1
  empeoran a 0.70-0.79: la solución usa valores graduados.

**Interpretación:** apoya la lectura 2 (la tarea no impone la química real
en este modelo) y ordena las hipótesis: el modelo simplificado no reproduce
con los signos reales la función biológica, así que el fallo no es de
"recuperación por el entrenamiento". Matiz: no prueba que el circuito real
no integre rumbo; el modelo omite ring neurons y fan-shaped body, inyecta la
velocidad angular a mano y fija ganancia y normalización. Hallazgo lateral:
la solución entrenada depende de magnitud efectiva graduada, así que
"magnitud fija" se cumple solo a medias.

**Siguiente paso:** la variante costosa del control de potencia (entrenar
sobre signos conocidos) pierde prioridad: sin un modelo cuya verdad
resuelva la tarea no es informativa con esta tarea. Queda la validación de
`ring_angle` (Hulse 2021, Fig. 10): si `ring_angle` es incorrecto, el
decodificador podría ser la causa de que los signos reales den pérdida ≈1.

## 2026-09-19 (4) — Corrección: referencia de "azar" errónea, `ring_angle` y prueba de signos reales

**Qué se hizo:** búsqueda del mapeo glomérulo-cuña en Hulse et al. (2021)
(texto de eLife; la Fig. 10 no es extraíble como tabla) y repetición de la
prueba de signos reales (entrada (3)) con `src/cx_net/real_sign_offset_check.py`:
pérdida invariante al desfase constante, tres mapeos de `ring_angle`, 150
asignaciones barajadas por mapeo. Salida: `data/interim/real_sign_offset_check.json`.

**Qué dice el artículo:** 16 cuñas del EB alternan entre PB izquierdo y
derecho; cada hemisferio muestrea el anillo completo a ≈45° (8 glomérulos),
L y R desfasados 22.5° (con el bump en L5, el otro queda entre R5 y R4); EPGt
(glomérulo 9) ≈ glomérulo 1. El mapeo del proyecto (L en media circunferencia,
R en la otra) lo contradice. El intercalado descartado el 2026-09-16 es el
conforme; el argumento de "continuidad dentro del hemisferio" de aquella
reversión no se sostiene (dentro de cada hemisferio quedan consecutivos a
45°). No verificado: el sentido de giro.

**Errores propios detectados (afectan a lo ya escrito):**
1. La referencia "azar ≈0.85" (y "held-out 0.6 = mejora real") compara con una
   fase aleatoria (≈1.0). Un decodificador constante en 0 da **0.179** en el
   held-out (hold_prob=0.3): los modelos entrenados (0.58-0.72) son 3-4 veces
   peores que la solución trivial. Un integrador perfecto con fase inicial
   arbitraria da 0.999: la tarea pide rumbo absoluto sin pista de dónde está
   el 0.
2. La entrada (3) (signos reales: 1.006 vs. barajados 0.983) no era una prueba
   justa por el punto anterior. El commit d1b2e05 contiene esa lectura.

**Resultado (pérdida invariante al desfase; decodificador constante = 0.060):**

| mapeo | signos reales | barajados (media±std, p1) | p(barajado ≤ real) |
|---|---|---|---|
| actual | 0.459 | 0.155±0.100, 0.064 | 0.98 |
| interl+ (R=L+22.5°) | 0.139 | 0.130±0.077, 0.066 | 0.77 |
| interl- | 0.115 | 0.125±0.057, 0.063 | 0.55 |

Modelos entrenados (mapeo actual): 0.08-0.36. Ninguno baja claramente del
suelo trivial. La conclusión "los signos reales no dan mejor tarea que los
barajados" se mantiene con la métrica justa y con los tres mapeos, pero ahora
es un hallazgo sobre el montaje: la red no integra rumbo ni con signos
entrenados ni con reales.

**Siguiente paso (decisión pendiente):** el resultado nulo de H1 informa poco
sobre la biología mientras la tarea sea resoluble trivialmente. Opciones:
(a) rediseñar la tarea con pista de fase inicial y pérdida invariante o
penalización de solución constante, con `ring_angle` intercalado, y
reentrenar; (b) cerrar el preprint como informe metodológico de estas
limitaciones. (a) reabre la fase experimental cerrada el 2026-09-18 (4).

## 2026-09-19 (5) — Fase real de cada neurona en el EB, medida desde coordenadas de sinapsis

**Contexto:** se elige la opción (a) de la entrada (4): rediseñar la tarea y
reentrenar (sin prisa de publicar). Bloqueo: el grafo solo tiene bodyId, type
e instance; el mapeo glomérulo -> fase no se puede validar. Comprobación por
topología (sin signos): con el mapeo intercalado los PEN de L se desplazaban
-24° y los de R -2° (esperable: ±45° opuestos), y los EPG de entrada y de
salida de cada PEN eran casi el mismo conjunto de glomérulos.

**Qué se hizo:** el token de neuPrint ya estaba en `.env` (no hace falta
pasarlo). MaleCNS no tiene ROI de cuñas (solo zonas radiales EBr*), pero
`fetch_synapses` da coordenadas 3D. `src/cx_net/extract_eb_angles.py`: 261.546
sinapsis en el EB de 110 neuronas (EPG, EPGt, PEN_a, PEN_b, PEG); plano por PCA
(89% de la varianza en 2 componentes); ángulo por sinapsis = atan2 en el plano;
ángulo por neurona = media circular ponderada por confianza. Salida
`data/raw/malecns/eb_angles.csv` (fuera del repo, regenerable; verificado
reproducible, dif. 5e-10 rad). `graph_utils.load_cx_graph(ring_source=
"eb_synapses", ring_sign=±1)`; el defecto sigue siendo el mapeo antiguo.

**Resultado:**
- Concentración angular por neurona 0.89-0.99: cada neurona ocupa una cuña.
- EPG por glomérulo (grados, aprox.): L1..L8 = 137, 177, 216, 265, 313, 356,
  47, 90 (sentido creciente, ~45° por glomérulo); R1..R8 = 115, 65, 22, 335,
  291, 239, 194, 156 (sentido DECRECIENTE). Cada hemisferio cubre el anillo
  completo; homólogos L/R desfasados ~22°. EPGt (glomérulo 9): 113-148°, en la
  fase del glomérulo 1, como en Hulse et al. (2021).
- **Corrección a la entrada (4):** el mapeo intercalado "conforme al artículo"
  tampoco era correcto: L y R van en sentidos opuestos (espejo); el intercalado
  supuso el mismo sentido en ambos. Eso explica el desplazamiento PEN
  incoherente. Con los ángulos medidos, los PEN de L y R se desplazan en
  sentidos opuestos con consistencia perfecta (L -6°, R +5.5° con signo +1;
  concentración 1.0). La magnitud (~6°, no ~45°) es probablemente por sinapsis
  EPG<->PEN recíprocas en el EB que el grafo no distingue de las del PB; el
  sentido sí es informativo.
- Sentido de giro (`ring_sign`): el método lo deja arbitrario. Regla por
  topología, sin NT: que los PEN de L (empujados con av>0) se desplacen en
  sentido +, es decir `ring_sign=-1`. Evidencia débil (magnitud pequeña): en el
  piloto se probarán ambos sentidos y se decidirá por desempeño en la tarea
  (nunca por acuerdo con el neurotransmisor); queda registrado como grado de
  libertad.

**Siguiente paso:** tarea anclada (fase inicial aleatoria inyectada como pista
breve en EPG/EPGt, objetivo theta0 + integral de la velocidad), con velocidad
angular suficiente para que "recordar theta0 sin integrar" no sea una solución
buena; línea base trivial a reportar siempre; piloto de 1 semilla por sentido
antes de barridos.

## 2026-09-19 (6) — Piloto de la tarea anclada: la red ancla la fase pero no integra

**Qué se hizo:** piloto de 1 semilla por sentido de giro (`ring_sign=+1/-1`),
`anchor=True`, `max_av=0.15`, `hold_prob=0.3`, `sign_reg=0.05`, 1000 épocas
(`data/interim/anchor_pilot_{p,m}.log`, `model_anchor_pilot_{p,m}.pt`). Criterio
fijado ANTES del piloto: bajar claramente de la referencia "recordar theta0 sin
integrar" para lanzar barridos.

**Resultado (held-out fijo, 30 ensayos):**

| | held-out |
|---|---|
| recordar theta0 sin integrar (referencia exacta) | 0.466 |
| decodificador constante | 1.105 |
| piloto ring_sign=+1 | 0.452 |
| piloto ring_sign=-1 | 0.477 |

Error de anclaje a t=20: 7-10°. Pendiente (cambio decodificado)/(cambio real)
≈ 0.01 y -0.03; recorrido máximo del bump 10-20° mientras el rumbo real cambia
~65°. Es decir: ancla y mantiene la fase (mucho mejor que constante), no
integra. **Criterio no cumplido: no se lanzan barridos ni H1.** Polarización
alta igualmente (mean_abs_sign 0.76-0.79).

**Sonda de dinámica** (velocidad constante av=0.3 desde t=25, ganancia de
entrada 3/10/30; grados que se mueve el bump; solo diagnóstico):
- Entrenados: |mov| <= 30° y decreciente con la ganancia.
- Signos reales fijos: ±188° con ganancia 10 y 30 (7° con 3), el signo sigue a
  `ring_sign`; barajados: 17-88°.
- La ganancia 10 -> 30 no cambia nada y `max r` = 0.99-1.0 en todos los casos:
  los EPG están saturados (tanh, `recurrent_gain=4`). El bump se desplaza a un
  punto fijo y se para, no se mueve con velocidad proporcional a la entrada.
  Régimen saturado, no un problema de optimización.
- Con `ring_sign=+1` los signos reales empujan el bump en el sentido de av>0;
  la regla por topología (entrada (5), evidencia débil) decía -1. Sin resolver.

**Interpretación (hipótesis, sin verificar):** con `tanh` saturado y
`recurrent_gain=4` la dinámica no tiene un régimen casi lineal donde la
velocidad del bump escale con la entrada, así que ningún signo entrenado
alcanza a integrar. La saturación explica también por qué los intentos previos
de hiperparámetros (`tau`, `recurrent_gain`) no mostraron efecto: la tarea
antigua no exigía integrar.

**Siguiente paso (decisión pendiente):** explorar `recurrent_gain`, `tau` y
ganancia de entrada buscando un régimen donde algún modelo integre. Riesgo
metodológico: si el criterio es "los signos reales integran", se elige el
régimen con ayuda del neurotransmisor (no filtra información al aprendiz, que
solo ve hiperparámetros, pero es un grado de libertad que hay que declarar).
Alternativa: elegirlo solo con redes entrenadas / barajadas.

## 2026-09-19 (7) — Búsqueda de régimen (criterio A): con signos reales la red no integra en ningún régimen probado

**Decisión previa (usuario):** criterio A -- elegir el régimen dinámico de modo
que la tarea sea realizable con los signos reales. El aprendiz nunca ve el
neurotransmisor (solo los hiperparámetros elegidos), pero el régimen se
selecciona con ayuda del ground truth: **grado de libertad que debe declararse
en el preprint.**

**Qué se hizo:** `src/cx_net/regime_search.py`, sin entrenar, signos reales
fijos, ambos sentidos de giro, conjunto de AJUSTE (semillas 800M, 20 ensayos;
el held-out 900M queda reservado). Métricas: pérdida absoluta y pendiente
(cambio decodificado / cambio real desde t=20; 1 = integra a la velocidad
correcta). Referencia del conjunto de ajuste: recordar theta0 = 0.537,
constante = 0.932. Tres rondas (la 2 y la 3 diseñadas tras ver la anterior):
1. `recurrent_gain` {0.5..16} x `tau` {2,5,10} x ganancia de entrada {1..30} x
   ganancia de pista {3,10} (144 configs por sentido).
2. `recurrent_gain` {0.75..3} x `tau` {10,20} x ganancia de entrada {3..300}
   (50 por sentido).
3. Activación rectificada `relu(tanh(x))` (tasas >= 0; opción nueva
   `CXRingNetwork(activation="rectified")`, defecto sin cambios) con
   `recurrent_gain` {1..16} x `tau` {2..20} x ganancia de entrada {1..100}
   (100 por sentido). Motivo: con `tanh` las tasas pueden ser negativas y una
   neurona inhibidora con tasa negativa excitaría a sus dianas.

**Resultado:**
- Ronda 1: mejor pérdida 0.493 (pendiente -0.08); las configs con pendiente
  0.6-1.4 (recurrencia débil, tau=2) tienen pérdida 0.76-0.93: siguen la
  velocidad pero pierden el ancla (el bump no persiste).
- Ronda 2: mejor pérdida 0.477 (pendiente 0.03); ninguna con pendiente 0.6-1.4.
- Ronda 3: mejor pérdida 0.470 (pendiente 0.05); la única con pendiente en
  rango tiene pérdida 1.02.
- En ninguna ronda hay una configuración que combine ancla (pérdida por
  debajo de la referencia) y pendiente ~1. Las mejores pérdidas (~0.47-0.49)
  son solo memoria con un beneficio marginal sobre 0.537.
- Tensión observada: mantener el bump exige recurrencia fuerte; moverlo con
  velocidad proporcional exige régimen casi lineal.

**Conclusión:** en este montaje (152 neuronas del núcleo, magnitudes fijas
normalizadas por neurona destino, dinámica de una sola constante de tiempo,
entrada de velocidad inyectada en PEN) la red con los signos reales NO
integra el rumbo en ningún régimen probado (~440 evaluaciones). Por tanto la
tarea no es realizable con la química real en este modelo y un H1 nulo no
informa sobre la biología. Se detiene la búsqueda: más rondas serían más
grados de libertad sin base.

**Hipótesis de por qué (sin verificar):** la normalización por neurona
destino borra la ganancia relativa entre entradas de tipos distintos (EPG,
PEN, Delta7) que en el circuito real sí importa; falta la entrada de ring
neurons/ExR; la entrada de velocidad sintética (antisimétrica por hemisferio,
en PEN) puede no ser el mecanismo real; dinámica de una sola tau.

**Siguiente paso (decisión pendiente):** (1) cerrar y reformular el preprint
como informe metodológico con esta comprobación de realizabilidad como
resultado central; (2) ampliar el modelo con ganancias por TIPO celular
entrenables (pocos parámetros; el signo por arista sigue siendo el único
parámetro de arista) y repetir la comprobación de realizabilidad con signos
reales, cuidando que las ganancias no se ajusten con los signos reales para
luego dárselas al aprendiz (filtraría información).

## 2026-09-19 (8) — Parámetros por tipo celular: con signos reales la tarea SÍ es realizable y discrimina la química

**Decisión previa (usuario):** último intento con la opción 2 de la entrada (7):
ganancias por tipo celular entrenables. Protocolo en dos pasos, fijado antes de
correr nada, para no filtrar información del neurotransmisor al aprendiz:
1. **Realizabilidad** (solo control): signos reales fijos, se entrenan SOLO los
   parámetros por tipo. Pregunta de sí/no; sus valores no se reutilizan.
2. **H1**: signos por arista y parámetros por tipo se entrenan conjuntamente
   desde un inicio neutro (ganancias 1, sesgos 0, signos ~0), sin nada del paso 1.

**Qué se añadió** (`model.py`, `learn_type_params=True`): ganancia positiva por
cada par de tipos origen->destino (6x6, `exp(log_pair_gain)`), un sesgo por tipo y
una escala global de la entrada: ~43 parámetros compartidos por tipo, nunca por
arista. `train()`: `activation`, `type_params`, `in_gain`, `cue_gain`,
`real_sign_control`, `control_shuffle_seed`. `integration_diagnostics`: pérdida,
error de anclaje y pendiente. Hiperparámetros fijos: `recurrent_gain=2`, `tau=10`,
`in_gain=10`, `cue_gain=10` (zona de las mejores pérdidas de la búsqueda (7), elegida
con ayuda de los signos reales -> **grado de libertad declarable**), 600 épocas.

**Aviso de entorno:** el `python` del sistema tiene torch 2.4.1, donde el gradiente
de `atan2(0,0)` es `nan`; el `.venv` (torch 2.14) da 0. Los entrenamientos por
defecto DEBEN usar `.venv/Scripts/python.exe` (verificado: reproduce exactamente
epoch 0/25 de `signreg05_seed0`). Los pilotos anclados de la entrada (6) usaron el
Python del sistema; la pista deja a los EPG != 0 en t=0, así que no fue afectado.

**Resultado paso 1 (held-out fijo, referencia "recordar theta0" = 0.466):**

| variante | held-out | pendiente | anclaje |
|---|---|---|---|
| tanh, ring_sign=+1, signos reales | 0.108 | 0.82 | 4° |
| tanh, ring_sign=-1, signos reales | 0.125 | 0.87 | 7° |
| rectificada, ring_sign=-1, signos reales | 0.172 | 0.73 | 4° |
| rectificada, ring_sign=+1, signos reales | 0.467 | 0.00 | 3° |

**Control decisivo: signos barajados** (mismo recuento 110/42, 4 permutaciones por
sentido, tanh, idéntico protocolo): held-out 0.450-0.598 (ring_sign=+1) y
0.430-0.469 (ring_sign=-1), pendiente entre -0.05 y 0.20: todos en "memoria sin
integrar". Los 2 con signos reales: 0.108 y 0.125.

**Conclusión:** con ganancias por tipo entrenables, la red con la química real
integra el rumbo y la red con neurotransmisores barajados no. La tarea es
realizable Y discrimina la química (8/8 barajados fallan, 2/2 reales funcionan,
ambos sentidos de giro). Por primera vez un H1 nulo sería informativo y un H1
positivo, interpretable. Ambos sentidos de giro funcionan igual con tanh, así que
`ring_sign` deja de ser un grado de libertad relevante para esta conclusión.
Límite: los parámetros por tipo se ajustaron CON los signos reales, así que esto
prueba realizabilidad, no que un aprendiz de signos los encuentre.

**Siguiente paso:** paso 2. Piloto de 1 semilla, 4 variantes (ring_sign +-1 x
sign_reg {0.05, 0}), 1000 épocas, `type_params=True`, inicio neutro
(`joint_sr*_{p,m}`). Criterio: held-out claramente por debajo de 0.466 con
pendiente ~1. Si el aprendiz no encuentra la solución, el resultado es "el
descenso de gradiente no recupera una solución que existe", distinto de "la
tarea no la impone".

## 2026-09-19 (9) — Paso 2: el aprendiz de signos no encuentra la solución que existe

**Qué se hizo:** piloto de 1 semilla, 4 variantes (ring_sign +-1 x sign_reg
{0.05, 0}), 1000 épocas, signos por arista + parámetros por tipo entrenados
conjuntamente desde inicio neutro (ganancias 1, sesgos 0, signos ~N(0, 0.1)),
`recurrent_gain=2`, `tau=10`, `in_gain=10`, `cue_gain=10`, tanh, `hold_prob=0.3`,
`max_av=0.15`. Etiquetas `joint_sr{0.05,0.0}_{p,m}`, `.venv`.

**Resultado (held-out fijo; referencia memoria pura 0.466; solución con signos
reales + ganancias por tipo: 0.108/0.125):**

| sign_reg | ring_sign | held-out | pendiente | anclaje | mean_abs_sign |
|---|---|---|---|---|---|
| 0.05 | +1 | 0.499 | -0.01 | 4° | 0.71 |
| 0.05 | -1 | 0.477 | -0.01 | 4° | 0.77 |
| 0.0 | +1 | 0.561 | -0.10 | 6° | 0.47 |
| 0.0 | -1 | 0.458 | 0.00 | 4° | 0.56 |

La EMA de entrenamiento quedó plana en ~0.53-0.56 desde la época ~100 en las
cuatro (los "mejores lotes" 0.23-0.25 son ruido de selección). **Ninguna integra.**

**Interpretación:** hay una solución en el espacio de búsqueda (entrada (8): signos
reales + ganancias por tipo, 0.11) y la tarea la distingue de las alternativas
barajadas, pero el descenso de gradiente desde un inicio neutro converge al
óptimo local "memoria sin integrar". Hallazgo de APRENDIBILIDAD, distinto de
"la tarea no la impone". Hipótesis sin verificar: paisaje de pérdida con un
plateau ancho entre la solución de memoria y la de integración (BPTT de 200
pasos con dinámica casi saturada); el aprendiz de signos parte de |tanh|~0.1
y el gradiente hacia la integración es débil; no se probó ningún currículo, otro
inicio, otra tasa de aprendizaje ni más épocas (cada una sería un grado de
libertad más; el usuario fijó esta como última tentativa).

**Lectura para el preprint:** con los tres controles juntos (entradas (7)-(9)):
(i) la química real supera claramente a la barajada en resolver la tarea con
la topología real (2/2 reales integran; 0/8 barajados; ganancias por tipo
optimizadas igual en todos); (ii) un aprendiz de signos genérico no recupera
esa solución. Un H1 "el aprendiz converge a la química real" no es evaluable
porque el aprendiz no converge a NINGUNA solución que integre. Lo que sí se
sostiene es H1 en versión de suficiencia funcional: la química real es una
solución (y las barajadas no lo son) -- con n pequeño (4 barajados por sentido de
giro; p de rango 1/5 por sentido).

**Siguiente paso (decisión pendiente):** ver mensaje al usuario. Opciones:
(1) cerrar aquí y reescribir el preprint con el resultado de suficiencia
funcional + fallo de aprendibilidad + trampas metodológicas; (2) reforzar (i)
con más barajados (p. ej. 20) y una curva dosis-respuesta de neuronas
intercambiadas (10/25/50%); (3) atacar la aprendibilidad (currículo, inicio, lr).

## 2026-09-19 (10) — Refuerzo de la suficiencia funcional: más barajados y dosis-respuesta (DISEÑO, fijado antes de ver resultados)

**Motivo:** tras la entrada (9), el resultado que sí se sostiene es de suficiencia
funcional (signos reales + ganancias por tipo integran; barajados no), pero con n
pequeño (4 barajados por sentido de giro; p de rango = 1/5). Se refuerza con más
barajados y una curva dosis-respuesta.

**Diseño** (todo `ring_sign=+1`, tanh, `recurrent_gain=2`, `tau=10`, `in_gain=10`,
`cue_gain=10`, `hold_prob=0.3`, `max_av=0.15`, 600 épocas, `real_sign_control=True`
con `type_params=True`, mismo protocolo que la entrada (8); 19 corridas nuevas en
paralelo, `data/interim/launch_dose.sh`):
- 2 réplicas más con signos reales (`seed=1,2`; ya existe `seed=0`: `realctl_tanh_p`).
- 8 barajados más (`control_shuffle_seed=5..12`; ya existen 1-4 -> 12 en total).
- Intercambio parcial: fracción 0.10, 0.25, 0.50 de neuronas cuyas etiquetas se
  permutan entre sí, 3 réplicas por nivel (`control_swap_fraction`,
  `control_shuffle_seed=1..3`). Al permutar dentro del subconjunto solo cambia la
  etiqueta de las neuronas que reciben la otra: se registran `frac_nodes_changed`
  y `frac_edges_changed` en el JSON y ESA es la abscisa de la curva, no la fracción
  nominal.

**Criterio primario (fijado antes de ver resultados):** pérdida held-out (30 ensayos
fijos, semillas 900M). Secundarios: pendiente e error de anclaje. Estadístico: p =
(1 + #{barajados con held-out <= media de las réplicas reales}) / (N_barajados + 1).
"Integra" = held-out < 0.466 (memoria sin integrar) Y pendiente > 0.5. No se
excluye ninguna corrida a posteriori; se reportan todas. La curva dosis-respuesta
es descriptiva (sin test de tendencia preespecificado).

**Riesgos declarados:** (a) el sentido de giro está fijado a +1 (con -1 el resultado
de la entrada (8) fue equivalente); (b) las ganancias por tipo se optimizan en cada
corrida con las etiquetas de esa corrida, así que la comparación real-vs-barajado
es de "mejor ajuste alcanzable", no de un ajuste único compartido; (c) el
`seed` de entrenamiento coincide (0) entre barajados, salvo las réplicas reales.

## 2026-09-19 (11) — Resultados del refuerzo; el neurotransmisor está determinado por el tipo celular

**Resultados** (diseño y criterio de la entrada (10); `src/cx_net/analyze_dose.py`,
`data/interim/dose_analysis.json`; ninguna corrida excluida):

| grupo | n | held-out | pendiente | integran (<0.466 y pendiente>0.5) |
|---|---|---|---|---|
| signos reales (seeds 0,1,2) | 3 | 0.108, 0.124, 0.106 | 0.82, 0.75, 0.84 | 3/3 |
| barajados (aristas cambiadas 35-50%) | 12 | 0.444-0.598, media 0.496 | max 0.04 | 0/12 |
| intercambio 10% nominal (aristas cambiadas 3.9-4.4%) | 3 | 0.474-0.507 | -0.03 a 0.01 | 0/3 |
| intercambio 25% (7.5-11%) | 3 | 0.463-0.491 | -0.01 a 0.01 | 0/3 |
| intercambio 50% (18-23%) | 3 | 0.475-0.491 | -0.02 a 0.07 | 0/3 |

- **Estadístico preespecificado:** p = (1 + #{barajados <= media real}) / (12 + 1) =
  **0.077**, que es el mínimo alcanzable con 12 barajados (1/13). NO llega a <0.05;
  harían falta >= 19. **Aclaración importante:** las 3 corridas reales comparten
  UNA sola asignación de etiquetas (difieren solo en la semilla de entrenamiento);
  demuestran reproducibilidad del entrenamiento, no 3 asignaciones distintas. Un
  test exacto "los 3 reales ocupan los 3 primeros puestos de 15" (p = 1/455) sería
  incorrecto y NO se usa.
- **Dosis-respuesta abrupta, no gradual:** con solo ~4% de aristas cambiadas
  (~6 neuronas) ya no integra ninguna de las 3 réplicas. La solución es muy frágil.
  Cautela: puede reflejar que el circuito real esté finamente ajustado en este
  modelo O que el optimizador de 43 parámetros por tipo no pueda compensar
  perturbaciones pequeñas (sensibilidad del paisaje); no se distinguen.

**Hallazgo estructural (comprobado):** en este subcircuito el neurotransmisor
está determinado por completo por el tipo celular:

| tipo | neuronas | neurotransmisor |
|---|---|---|
| Delta7 | 42 | glutamato (100%) |
| EPG, EPGt, PEG, PEN_a, PEN_b | 46+4+18+20+22 = 110 | acetilcolina (100%) |

El cuaderno lo sabía (entradas de diseño: "Delta7=glutamato, resto=acetilcolina"),
pero no se extrajeron las consecuencias:
1. La "química real" son ~6 bits a nivel de tipo, en la práctica UNO: "Delta7
   inhibe, el resto excita" (el motivo clásico de inhibición lateral del ring
   attractor). Aristas con origen Delta7: 2846 de 9160.
2. **El control positivo de potencia (entrada 2026-09-19 (2)) sobreestima la
   potencia:** sembró señal por neurona (152 unidades independientes) o por
   arista; una señal real estaría estructurada por tipo (6 unidades, una de ellas
   con 42 neuronas). Las cifras "80% de detección a +3.3 pp" no son aplicables a
   una señal a nivel de tipo y deben rehacerse (o retirarse) en el borrador.
3. Barajar etiquetas entre neuronas rompe la estructura por tipo; el espacio de
   hipótesis natural a nivel de tipo es 2^6 = 64 patrones de signo por tipo, y
   la asignación real es uno de ellos. Con enumeración exhaustiva el p-valor
   sería exacto (rango entre 64), sin muestreo.
4. Los "barajados" son alternativas muy heterogéneas (35-50% de aristas con signo
   distinto); no responde qué parte de la estructura real es necesaria.

**Siguiente paso (decisión pendiente):** enumerar los 64 patrones de signo por
tipo (mismo protocolo, 600 épocas, ~2.5 h con ~20 corridas en paralelo) para saber
cuáles integran; con eso el resultado de suficiencia funcional queda exacto y se
ve qué tipos importan (¿basta "Delta7 inhibe"?, ¿importa el signo de los PEN?).
Después, reescribir el borrador con estos resultados y rehacer/retirar el control
de potencia.

## Plantilla para próximas entradas

```
## AAAA-MM-DD — <título breve>

**Qué se hizo:**
**Por qué:**
**Resultado / número clave:**
**Siguiente paso:**
```

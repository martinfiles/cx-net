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
integra el rumbo en ningún régimen probado (588 evaluaciones (288+100+200; corrección de un recuento previo erróneo de ~440)). Por tanto la
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

## 2026-09-20 — Enumeración de los 64 patrones de signo por tipo celular (DISEÑO, fijado antes de ver resultados)

**Motivo:** entrada (11): el neurotransmisor es función exacta del tipo (Delta7 =
glutamato, resto = acetilcolina), así que la hipótesis natural es un signo por
tipo: 2^6 = 64 patrones. Enumerarlos todos da un p-valor exacto (rango de la
asignación real entre los 64, sin muestreo) y muestra qué tipos importan.

**Diseño:** `control_type_mask` en `train()` (máscara de 6 bits; bit i = 1 ->
el tipo `TYPE_ORDER[i]` inhibe; orden Delta7, EPG, EPGt, PEG, PEN_a(PEN1),
PEN_b(PEN2); el signo se aplica a todas las aristas SALIENTES del tipo).
Máscara 1 (solo Delta7 inhibe) == neurotransmisor real (verificado). 63 corridas
nuevas (`types_mask{0..63 salvo 1}_p`, `data/interim/launch_types.sh`, 21 en
paralelo); la máscara 1 se reutiliza de `realctl_tanh_p` (seed 0). Mismo
protocolo que las entradas (8)/(10): `ring_sign=+1`, tanh, `recurrent_gain=2`,
`tau=10`, `in_gain=10`, `cue_gain=10`, `hold_prob=0.3`, `max_av=0.15`, 600
épocas, `seed=0`, ganancias/sesgos por tipo entrenables con signos fijos.

**Criterio primario (fijado antes):** "integra" = held-out < 0.466 Y pendiente
> 0.5. p exacto = #{patrones con held-out <= el de la máscara 1} / 64 (la real
incluida; si es la única que integra y la mejor, p = 1/64 = 0.0156). Descriptivos:
lista de patrones que integran, y para cada tipo la fracción de patrones que
integran con ese tipo excitador vs. inhibidor. No se excluye ninguna corrida.

**Riesgos declarados:** (a) una corrida por patrón y semilla 0: un patrón
"no integra" puede ser mala suerte de optimización, no imposibilidad; se
valorará con las réplicas de la real (0.106-0.124, muy estables); (b) los signos
son por tipo de ORIGEN de la arista; (c) `ring_sign=+1` fijo; (d) los
hiperparámetros dinámicos se eligieron con ayuda de la asignación real
(grado de libertad ya declarado en (7)/(8)).

## 2026-09-20 (2) — Resultado de los 64 patrones: la asignación real NO es especial; las tasas negativas del tanh abren un grado de libertad no fisiológico

**Resultado** (diseño y criterio de la entrada (12); `src/cx_net/analyze_types.py`,
`data/interim/types_analysis.json`; ninguna corrida excluida, 1 corrida por patrón, seed 0):
- **26 de 64 patrones de signo por tipo integran** (held-out < 0.466 y pendiente > 0.5).
- La asignación real (máscara 1, solo Delta7 inhibe): held-out 0.108, pendiente 0.82,
  **puesto 22 de 64 por held-out. p exacto = 22/64 = 0.344.** No hay evidencia de que
  sea especial.
- Mejores: máscaras 39/43/47 (0.054-0.055; incluyen D7 inhibidor), pero también la 2
  ("solo EPG inhibe": 0.063) y la 6 ("EPG+EPGt inhiben": 0.063), biológicamente absurdas.
- Fracción de patrones que integran según el signo del tipo: D7 exc 7/32, inh 19/32;
  EPG 10/32 vs 16/32; EPGt 14/32 vs 12/32; PEG 12/32 vs 14/32; PEN_a 14/32 vs 12/32;
  PEN_b 20/32 vs 6/32. Descriptivo, sin test; el único contraste marcado es PEN_b
  excitador, y D7 inhibidor ayuda pero no es necesario ni suficiente. No integrados:
  held-out 0.262-0.703 (mediana 0.474).

**Consecuencia:** RETRACTAR la lectura de "suficiencia funcional" de la entrada (9):
que 12 barajados por neurona fallen NO muestra que la química real sea especial,
sino que asignaciones que rompen la coherencia por tipo fallan. A nivel de tipo, 26 de
64 patrones funcionan, y la real es uno más (puesto 22).

**Mecanismo probable (verificado descriptivamente, 5 ensayos, t>=40):** con `tanh` las
tasas van en (-1, 1). En la red con la asignación real, las tasas de Delta7 son
negativas el 89% del tiempo (fracción >0: 0.11; media -0.17): una neurona
"inhibidora" con actividad negativa EXCITA a sus dianas. En "solo EPG inhibe" las EPG
oscilan ~50/50 alrededor de 0; en el patrón sin inhibición (que no integra) Delta7
tiene fracción >0 = 1.00 (media +0.98). Es decir, la solución explota tasas
negativas, una libertad no fisiológica que actúa como simetría de gauge y hace que
muchos patrones de signo sean funcionalmente equivalentes. Esto vale también para
la solución con la asignación real de la entrada (8): no es una solución biológica.
(Sin verificar formalmente la simetría de gauge; solo la observación descriptiva.)

**Implicaciones:** (1) El H1 "el aprendiz converge a la química real" sigue sin ser
evaluable (el aprendiz no aprende, entrada (9)). (2) La versión de suficiencia
funcional no se sostiene con tasas con signo. (3) Un test biológicamente válido
requiere tasas no negativas; la activación rectificada ya existe (`activation=
"rectified"`) pero en la entrada (8) fue inestable (real: 0.172 con ring_sign=-1,
0.467 con +1). Alternativas sin ReLU muerta: sigmoide/softplus (con tasa basal).
(4) Siguen siendo hallazgos sólidos: la referencia de azar errónea, la tarea sin ancla,
la geometría del anillo medida en sinapsis, la realizabilidad con parámetros por tipo,
el fallo de aprendibilidad y este sesgo de las tasas con signo.

**Siguiente paso (decisión pendiente):** (A) repetir la comprobación de realizabilidad y
la enumeración de 64 patrones con tasas no negativas (sigmoide o rectificada estable);
~1 h de comprobación + ~3 h de enumeración. Si la real destaca, es el resultado
biológico; si no, es un negativo definitivo. (B) Cerrar y reescribir el preprint con
lo anterior como informe metodológico/negativo, retirando el control de potencia por
neurona y la lectura de suficiencia.

## 2026-09-20 (3) — Sigmoide, validación mecanística, revisión crítica y datos

**Comprobación con tasas no negativas (sigmoide, `activation="sigmoid"`, tasa basal 0.5;
mismos hiperparámetros y 600 épocas; `sigm_*`):** signos reales `ring_sign=-1`: held-out
0.303, pendiente 0.33; `+1`: 0.470, pendiente 0.01; 4 barajados (+1): 0.536-0.955,
pendiente máx 0.27. Los reales quedan por delante de los barajados pero ninguno cumple
"integra" (< 0.466 y pendiente > 0.5): no concluyente; no se enumeran los 64 patrones (haría
falta una búsqueda de régimen propia).

**Validación mecanística (`src/cx_net/bump_check.py`, `results/bump_check.json`):** con velocidad
constante GRANDE (±0.3, 180 pasos) ningún modelo desplaza el bump (velocidad decodificada ≈ 0;
fuera del rango de entrenamiento: 54 rad). Con velocidad constante en rango (±0.01 a ±0.08 por
paso durante 90 pasos), desplazamiento decodificado frente a esperado: real (solo Delta7
inhibe) ganancia 0.54, R² 0.84 (esperado ±7.2 rad -> -2.8/+3.3, saturante); "solo EPG inhibe"
0.76 (R² 0.93); "EPG+EPGt inhiben" 0.75 (R² 0.92); sin inhibición 0.00 (desplazamiento
≈ -0.2 rad plano). Localización del bump |sum r e^{i th}|/sum|r| (EPG/EPGt): 0.62 real,
0.33-0.37 "solo EPG"/"EPG+EPGt", 0.64 sin inhibición. Conclusión: las redes que "integran" no son
un artefacto del decodificador (respuesta antisimétrica y aproximadamente proporcional), pero
son integradores COMPRESIVOS (ganancia < 1, saturantes), no perfectos; la "pendiente" de las
entradas anteriores mide integración en el rango de entrenamiento.

**Revisión crítica interna (`docs/expert-review.md`):** ocho preocupaciones principales. Cambios
aplicados al borrador: (M1) H1 se reduce a un bit conocido (Delta7 inhibe); (M2) "ground truth"
-> neurotransmisor anotado (predicción), supuestos de Dale/GluCl/receptor explícitos; (M3) "tasas
negativas no fisiológicas" retirado -> desviaciones de una basal nula y falta de
identificabilidad del signo; (M4) validación mecanística parcial; (M5) estado exploratorio,
umbral binario arbitrario, marginales no independientes, sesgo de selección de hiperparámetros a
favor de la real (que refuerza el nulo); (M6) resolución mínima del test por tipo (1/64; 1/6);
(M7) geometría "no es un hallazgo nuevo"; (M8) subsección de posicionamiento con referencias por
verificar. Abierto: replicar patrones con varias semillas, perfil/anchura del bump y comparación
con modelos de anillo ajustados a mano, validación del mapeo de fase en hemibrain, revisión
por alguien del campo.

**Datos y licencia:** MaleCNS (`male-cns:v1.0`, liberado 2026-06-08) está bajo CC-BY 4.0; se piden
citas. Referencia: Berg et al. (2026), "Sexual dimorphism in the complete Drosophila male central
nervous system connectome", Cell (publicado 2026-09-03); preprint bioRxiv 10.1101/2025.10.09.680999.
El repo no redistribuye datos; `results/` contiene derivados agregados (con atribución en el
README). Código bajo MIT.

**Figuras:** títulos matizados ("se asocian" en lugar de "pesan"), fig. 1 aclarada (tarea sin pista
de fase), fig. 5 ("por debajo de la basal").

## 2026-09-21 — Perfil del bump y kymographs (respuesta a la revisión externa)

**Qué se hizo:** `src/cx_net/bump_profile.py` -> `results/bump_profile.json` y figura 6
(kymographs, es/en). Para los mismos cuatro modelos de `bump_check.py`: perfil del bump
alineado con la fase decodificada (velocidad 0), ancho a media altura, contraste, fracción de
neuronas saturadas, y memoria de 8 fases de pista arbitrarias.
**Por qué:** el experto objetó que una pérdida baja no prueba un bump localizado; la validación
mecanística previa medía ganancia y localización global, no el perfil ni el movimiento en el tiempo.
**Resultado / número clave:** ancho 167° (real), 122°, 135° (integran) y 133° (no integra);
saturadas 0.00 / 0.00 / 0.00 / 0.83; error de memoria de fase a t=199: 0.40 / 0.30 / 0.28 / 0.18 rad.
Los tres que integran mueven una banda ancha y graduada; el que no integra es un patrón binario
congelado. La memoria de fase arbitraria NO discrimina (los cuatro la sostienen). Primera versión
del perfil sin restar la media poblacional daba un contraste engañoso (1.06) por las tasas con
signo; corregido antes de usarlo. Salvedades: una corrida por modelo, contenedores de 30°
(≈4 neuronas), sin comparación con un modelo de anillo a mano.
**Siguiente paso:** comparar el perfil con un anillo construido a mano (Kakaria-de Bivort /
Turner-Evans / Pisokas); barrido de escala global de pesos; replicar los 64 patrones con semillas.

## 2026-09-22 — Magnitudes libres por arista con signos fijos (propuesta del experto)

**Qué se hizo:** opción `edge_gain` (ganancia positiva por arista, `log_edge_gain`, init 0) en
`model.py`/`train.py`; `edge_gain_lr` para una tasa propia. Misma receta que los patrones por tipo
(600 épocas, tanh, ganancias por tipo). Ronda 1 (`edgegain_*`, misma tasa 0.05): real x3, 6 barajados
por neurona (los mismos que `realctl_shuf1..6`), mask0, mask2. Ronda 2 (`edgegain_lr005_*`, tasa 0.005
solo para las ganancias por arista): real x3, 6 barajados, mask0. Análisis: `analyze_edge_gain.py`.
**Por qué:** el experto sugirió fijar los signos y liberar solo magnitudes para probar que el problema
es «fuerza = conteo de sinapsis». Mi reserva de diseño (la original de `model.py`): un signo «equivocado»
se puede compensar con la magnitud; se midió la poda.
**Resultado / número clave:** ronda 1: real 0.470-0.475 (no integra), barajados 0.236-0.486 (1 de 6
integra), mask0 0.468, mask2 0.474; no hay poda (< 0.5 % de aristas de signo distinto). Ronda 2: real
0.103 / 0.078 / 0.083 (pendiente 0.80-0.91), barajados 0.353-0.578 (ninguno integra), mask0 0.464;
ganancias por arista entre 0.8 y 1.3 (p5-p95), sin poda. Lectura: ronda 1 = fallo de optimización
(la solución por tipo está contenida y aun así se pierde); ronda 2 = casi el resultado por tipo (mejora
0.02-0.04). No aísla el efecto de las magnitudes. La receta de la ronda 2 se eligió después de ver que
la real fallaba en la ronda 1 (sesgo a favor de la real, como en los demás hiperparámetros).
Un error de mi script de análisis (prefijo fijo) mostró primero de nuevo la ronda 1; se corrigió antes
de usar los números.
**Siguiente paso:** semillas 1 y 2 de los 63 patrones (en curso, `data/interim/launch_seeds.sh`);
no repetir `edge_gain` con tasas intermedias salvo que se quiera una curva dosis-respuesta de libertad.

## 2026-09-22 (2) — Replica de los 64 patrones con 3 semillas; nuevo ranking de la real

**Qué se hizo:** las 63 corridas restantes (todas menos la real, ya replicada en `dose_real_seed{1,2}_p`)
con semillas 1 y 2 (126 corridas, `data/interim/launch_seeds.sh`, tras la ronda de `edge_gain`).
Agregado en `results/types_analysis_3seeds.json` (media de held-out y voto de mayoría sobre
`integra` por patrón). Figuras 3 y 4 regeneradas con este archivo.
**Por qué:** responde a la pregunta pendiente del experto ("¿cuántas semillas por patrón antes de
creer un ranking?") y a la limitación ya anotada (M5 / sección 10) de una sola corrida por patrón.
**Resultado / número clave:** **28/64 integran por mayoría (antes 26/64); la real pasa del puesto
22 al 14 de 64 por held-out medio (p = 14/64 = 0.219, antes 0.344).** La real es estable: integra
en las 3 semillas (0.106, 0.124, 0.106). De los otros 63 patrones, 4 cambian de veredicto entre
semillas (mask14, mask15: de "no integra" a "integra"; mask45, mask57: al revés) — confirma que
una sola corrida es ruidosa. Ningún patrón cambia en sentido "integra con semilla 0 -> dejan de
integrar con más semillas" salvo mask45 y mask57. El sesgo se mantiene: la real sigue sin ser
especial (28/64 la superan o igualan).
**Siguiente paso:** ninguno pendiente de este bloque. Falta decidir si vale la pena una 4ª-5ª
semilla para los patrones que siguen cerca del umbral (mask14, mask15, mask45, mask57).

## 2026-09-22 (3) — Correccion de citas: verificacion de Pisokas, Turner-Evans, Franconville; "Green et al. 2017" no localizado

**Que se hizo:** verificacion de existencia (no de contenido completo) de las referencias de la
subseccion de posicionamiento: Pisokas, Heinze y Webb (2020, eLife 9:e53985); Turner-Evans et al.
(2020, Neuron 108:145); Franconville, Beron y Jayaraman (2018, eLife 7:e37017); Kakaria y de
Bivort (2017, Front. Behav. Neurosci. 11:8); Kim et al. (2017, Science 356:849); Prinz, Bucher y
Marder (2004, Nat. Neurosci. 7:1345). Las 6 existen con esos datos.
**Por que:** el experto no verifico ninguna cita en sus dos respuestas; hacia falta antes de
citarlas en una version publica.
**Resultado / numero clave:** las entradas (3) del 2026-09-16 y de mas abajo citan dos veces
"Green et al. (2017)" para Delta7 inhibidor / resto excitador en modelos analiticos del circuito.
No encuentro ningun articulo de Green con ese tema en 2017. Lo mas cercano es Green, Vijayan,
Mussells Pires, Adachi y Maimon (2019, Nat. Neurosci. 22:1460), sobre comparar el rumbo con un
objetivo interno, publicado 2 anos despues -- no confirma que sea la misma referencia mal
fechada ni que el contenido (Delta7 inhibidor) este ahi. **No usar "Green et al. 2017" en el
informe ni en el paquete de revision hasta verificarlo**; el hallazgo del ground truth (Delta7 =
glutamato inhibidor) no depende de esta cita, viene de neuPrint/Eckstein et al. (2024).
**Siguiente paso:** al escribir para bioRxiv, releer la entrada (3) del 2026-09-16 y buscar cual
articulo se queria citar (¿Turner-Evans et al. 2017 solamente? ¿otro Green?).

## 2026-09-28 — Correcciones tras la revisión externa (neurociencia computacional del sistema de dirección de cabeza)

**Qué se hizo:** se recibió una revisión externa completa (recalculó cifras desde `results/`,
reimplementó el generador de la tarea, regeneró las 6 figuras, verificó citas). Encontró 8
errores de cifra/texto y 7 problemas de fondo; veredicto: informe negativo honesto pero con
errores en el resultado principal y un hueco de reproducibilidad, publicable tras arreglarlos.
Se corrigieron los 8 primeros en `docs/preprint-discussion.md`, `docs/summary-en.md`, `README.md`
y `src/cx_net/make_figures.py` (figuras regeneradas):
1. "4 de 63 cambian de veredicto, 2 y 2" → falso: son 16/63 no unánimes, y de esos 4 difieren
   del voto de la semilla 0 (3 hacia "integra": mask14, 15, 57; 1 al revés: mask45).
2. "28/64" depende del criterio de agregación: 20 en las 3 semillas, 28 por mayoría, 36 en
   alguna. Ahora se da el rango, no un valor único.
3. El puesto de la real depende de la agregación: 14 (media), 19 (mediana), 28 (mejor semilla).
4. "Solo EPG inhibe: held-out 0.063" era la semilla 0; la media de las 3 es 0.238 (una semilla
   no converge). "EPG+EPGt inhiben" media 0.065, no 0.062.
5. "Green et al. 2017" existe (Nature 546:101); se retira la advertencia de la entrada
   2026-09-22 (3) y se añade la cita.
6. El desfase L/R de "≈22°" (fig. 2, resumen) es falso en general: las 16 posiciones se
   intercalan alrededor del anillo, paso medio ≈22.5°; el desfase Ln-Rn crece con n y solo
   vale ≈22° para n=1 y n=5. Corregido en el CSV-derivado, las figuras y el texto.
7. Texto desactualizado en las secciones 1, 10, 11 y 12 del borrador ("26 de 64" junto a
   "puesto 14"; bump sin caracterizar) y título de la figura 5 en inglés ("negative rates").
8. `results/runs_summary.csv` y `results/types_analysis_3seeds.json` /
   `types_explorer_3seeds.json` estaban desactualizados (139 filas, faltaban las 126 corridas
   de réplica y las 20 de `edge_gain`); no existía script para regenerar los dos últimos. Se
   creó `src/cx_net/analyze_types_3seeds.py` (lee `data/interim/results_*.json` ya guardados,
   sin reentrenar; reproduce los valores de `types_analysis_3seeds.json`; el JSON gana el campo `anchor_err_deg`, así que no es idéntico byte a byte; solo el fichero del explorador lo es) y se volvió a correr
   `collect_results.py` (139→286 filas). `requirements.txt` limpiado (quita `caveclient`,
   `navis`, `networkx`, `torch-geometric`, no usados; añade `matplotlib`, que sí se usa).

**Por qué:** todas las cifras y afirmaciones de texto de un informe negativo son parte de la
evidencia; un revisor externo detectó que varias no coincidían con los datos ya versionados en
`results/`, y que dos de los tres archivos que respaldan la cifra principal no eran
reproducibles con el código del repo.

**Resultado / número clave:** ninguna corrección cambia la conclusión cualitativa (la asignación
real integra pero no es especial). El punto B (grafo por compartimento PB/EB) queda pendiente:
requiere volver a consultar `fetch_adjacencies` por región en neuPrint (token pendiente). Se
verificó además, con los datos ya versionados, la interacción Delta7×EPG×PEN_b que propuso el
revisor (sección 6 del borrador): con PEN_b excitador, 0/8-8/8-8/8-6/8 según Delta7/EPG inhiban;
con PEN_b inhibidor solo 6/32 integran. Reformula el resultado como "un bucle EPG↔Delta7 con
signo neto negativo, más PEN_b excitador", no como ausencia de estructura.

**Siguiente paso:** (a) token de neuPrint para recalcular el desfase de PEN solo con aristas
PB→EB; (b) formalizar la simetría de signo y las clases de equivalencia por bucle (sección 11);
(c) 4ª-5ª semilla para mask14/15/45/57 y selección de checkpoint por EMA; (d) rediseñar la tarea
(velocidad correlada en el tiempo, varias vueltas, ruido de proceso) antes de reportar cualquier
cifra de "gana margen" entre patrones.

## 2026-09-28 (2) — Desfase de los PEN por compartimento (PB de entrada, EB de salida): el token de neuPrint confirma la sospecha del revisor

**Qué se hizo:** `src/cx_net/pen_offset_by_region.py` (nuevo). Usando el desglose por
`roi` que ya devuelve `fetch_adjacencies` (y que `build_topology_only_graph` descarta al
sumar todas las regiones), se recalculó el desfase de cada neurona PEN sin mezclar
compartimentos: fase de entrada = media circular ponderada de sus EPG/EPGt presinápticas
SOLO en PB; fase de salida = lo mismo con sus EPG/EPGt postsinápticas SOLO en EB.

**Por qué:** la revisión externa (2026-09-28) señaló que el desfase de ≈6° medido antes
(entrada 2026-09-19 (5)) es sospechosamente pequeño frente al anatómico (varias decenas
de grados), y que `fetch_adjacencies` ya da la tabla por región para comprobarlo.

**Resultado / número clave:** [CORREGIDO 2026-09-29: solo 1.439 de ≈12.3k pesos EPG→PEN caen en PB; el resto en EB] las sinapsis EPG→PEN caen en PB (1.439 de
pesos) y casi todas las PEN→EPG en EB (23.976 de pesos) — la separación por región tiene
sentido. Con eso, el desfase sale mucho mayor y consistente: PEN_a −54.1°(L) / +54.3°(R)
(n=10+10 neuronas con ambos lados medibles), PEN_b −51.4°(L) / +50.4°(R) (n=11+11) — cerca
de un glomérulo (≈45°), con la misma antisimetría L/R perfecta que el ≈6° anterior. La
fase propia de cada PEN (de sus propias sinapsis EB, independiente de la conectividad)
coincide con la fase de salida hasta ~1°. **Confirma la sospecha: sumar PB+EB diluía el
desfase real por un factor ≈9.** Salida: `results/pen_offset_by_region.json`.

**Siguiente paso:** reconstruir `graph_no_sign.csv` completo por compartimento (no solo
el desfase de fase) y reentrenar el modelo por tipo sobre ese grafo, para ver si un
mecanismo de rotación más fuerte cambia la necesidad de ganancias por tipo entrenables
(sección 6 del borrador).

## 2026-09-28 (3) — Grafo por compartimento (PB+EB): reconstruido, y la búsqueda de régimen sin entrenar da el mismo negativo

**Qué se hizo:** `src/cx_net/extract_graph_pb_eb.py` (nuevo) reconstruye
`data/raw/malecns_pb_eb/graph_no_sign.csv` filtrando `conn_df` a `roi in {PB, EB}` antes
de sumar pesos (mismas 152 neuronas y mismo `eb_angles.csv` que `data/raw/malecns/`, sin
recalcular). `train.py` y `regime_search.py` ganaron un parámetro `graph_data_dir` /
`graph_data_dir` (antes usaban `DATA_DIR` fijo) para poder apuntar a este grafo alternativo
sin tocar el comportamiento por defecto. Se repitieron las rondas 1 y 2 de la búsqueda de
régimen sin entrenar (336 evaluaciones, signos reales fijos, sin ganancias por tipo) sobre
el nuevo grafo.

**Por qué:** la entrada anterior (2026-09-28 (2)) mostró que el desfase de los PEN estaba
diluido ≈9× al mezclar regiones; el siguiente paso lógico es ver si un grafo sin diluir
cambia el resultado negativo de la sección 6 ("con signos reales fijos, ningún régimen
dinámico integra sin ganancias por tipo").

**Resultado / número clave:** peso total 141.014 → 122.158 (86.6% conservado; el resto son
sinapsis en NO, FB, LAL(L/R) y CentralBrain-unspecified). Con ese grafo, la mejor pérdida de
las 336 evaluaciones es 0.491-0.517 con pendiente entre −0.08 y +0.03 en las 10 mejores
configuraciones de cada ronda — el mismo negativo que con el grafo mezclado (0.470-0.493,
pendiente ≈0). **Un desfase de PEN anatómicamente correcto no basta por sí solo: hace falta
además reponderar las magnitudes (ahora mismo, conteo de sinapsis) para que ese mecanismo se
traduzca en integración.** Guardado en `results/regime_search_pbeb.csv` y
`results/regime_search_r2_pbeb.csv`.

**Siguiente paso:** lanzadas (en curso al escribir esto) 3 semillas reales + 4 barajados con
ganancias por tipo entrenables sobre el mismo grafo (`data/interim/launch_pbeb.sh`,
`pbeb_real_seed{0,1,2}_p`, `pbeb_shuf{5,6,7,8}_p`), misma receta que la sección 6
(`recurrent_gain=2, tau=10, in_gain=10, cue_gain=10`, 600 épocas). Compara si el held-out /
pendiente de la solución ENTRENADA cambia frente a 0.106-0.124 / 0.75-0.84 del grafo
mezclado.

## 2026-09-28 (4) — Con ganancias por tipo sobre el grafo PB+EB, la real va PEOR: resultado contrario a la hipótesis de partida

**Qué se hizo:** terminaron las 7 corridas de la entrada anterior. `collect_results.py`
vuelto a correr (293 filas, antes 286).

**Por qué:** cerrar la pregunta de la entrada (3): ¿un mecanismo de rotación sin diluir
(desfase de PEN ≈50°) hace que la asignación real integre mejor, o necesite menos ayuda de
las ganancias por tipo, que con el grafo mezclado?

**Resultado / número clave — inesperado, contrario a la hipótesis:** con los MISMOS
hiperparámetros que en la sección 6 (elegidos para el grafo mezclado, sin volver a
ajustar), la real va PEOR sobre el grafo PB+EB: held-out 0.281 / 0.299 / 0.339 (media
0.306; antes 0.106-0.124) y pendiente 0.42 / 0.40 / 0.36 (media 0.39; antes 0.75-0.84).
**Con el criterio de "integra" (held-out < 0.466 y pendiente > 0.5), NINGUNA de las 3
semillas integra** sobre este grafo (el held-out mejora respecto a la memoria pero la
pendiente no llega). Los 4 barajados tampoco integran (held-out 0.43-0.60, pendiente
−0.02 a +0.07). Quitar el 13.4% del peso fuera de PB/EB (NO, FB, LAL(L/R),
CentralBrain-unspecified) no "purifica" el mecanismo, lo debilita: esas sinapsis
aparentemente contribuyen dinámica útil, no solo dilución del desfase de fase. **La
hipótesis de la entrada 2026-09-28 (2) ("el grafo mezclado diluye y por tanto debilita
el mecanismo") queda parcialmente refutada por este experimento**: diluir el desfase de
fase (punto B del revisor) es real y se confirmó, pero no implica que restringir el grafo
mejore el entrenamiento — el efecto neto es el contrario, al menos con este régimen
dinámico.

**Salvedad importante, sin resolver:** los hiperparámetros dinámicos no se volvieron a
ajustar para este grafo (mismo "criterio A" de sesgo declarado que en la sección 6, pero
esta vez sin siquiera intentar favorecer a la real). No se puede distinguir todavía entre
"el núcleo PB+EB es dinámicamente insuficiente" y "hace falta un régimen distinto (quizás
`recurrent_gain` mayor, para compensar el 13.4% de recurrencia perdida) para este grafo
más pequeño".

**Siguiente paso:** repetir la búsqueda de régimen tunable (no solo la versión sin
entrenar de la entrada (3)) específicamente para el grafo PB+EB antes de sacar
conclusiones sobre si el núcleo PB+EB es insuficiente o solo necesita otro régimen.

## 2026-09-28 (5) — Búsqueda de régimen reajustada para PB+EB (recurrent_gain hasta 24): mismo negativo, [etiqueta «salvedad cerrada» retirada 2026-09-29]

**Qué se hizo:** nueva rejilla en `regime_search.py`
(`regime_search_pbeb_r3.csv`): `recurrent_gain` ∈ {1.5, 2, 3, 4, 6, 8, 12, 16, 24} (el
grafo mezclado solo llegaba a 16), `tau` ∈ {5, 10, 20}, `in_gain` ∈ {10, 30, 100, 300},
`cue_gain=10`; 216 evaluaciones sin entrenar, signos reales fijos, sobre
`data/raw/malecns_pb_eb`.

**Por qué:** cerrar la salvedad de la entrada (4): ¿la real entrenaba peor sobre PB+EB
porque el régimen dinámico (`recurrent_gain=2`) estaba mal ajustado para el 13.4% de
recurrencia perdida, o el núcleo PB+EB es insuficiente para cualquier régimen razonable?

**Resultado / número clave:** el mismo negativo, incluso empujando `recurrent_gain` a 24
(1.5× el máximo probado con el grafo completo): mejor pérdida 0.498 (`rec_gain=2, tau=5,
in_gain=100, ring_sign=-1`), pendiente −0.12 en esa configuración y entre −0.14 y +0.01
en las 20 mejores. **Ningún régimen explorado, ni siquiera con recurrencia muy alta,
logra que los signos reales solos integren sobre el grafo PB+EB** — el mismo patrón que
con el grafo mezclado (sección 6) y que con la rejilla anterior sobre este mismo grafo
(entrada (3)). Cierra la salvedad: no es un problema de `recurrent_gain` insuficiente;
el núcleo PB+EB, con las magnitudes fijas al conteo de sinapsis, no alcanza a integrar
en ningún punto de una rejilla de 388+216=604 evaluaciones combinadas (corregido 2026-09-29; antes 336/552). Sigue sin probarse
`tau` fuera de {5,10,20} ni `in_gain` > 300, pero el patrón (techo de pérdida ≈0.49-0.52,
pendiente ≈0 en todo el rango) ya está muy saturado como para esperar que un punto
aislado más allá cambie la conclusión cualitativa.

**Siguiente paso:** ninguno pendiente de este bloque concreto. La pregunta que sigue
abierta (sección 11 del borrador) es por qué las ganancias por tipo entrenadas SÍ
resuelven la tarea sobre el grafo mezclado (0.106-0.124) y fallan sobre PB+EB
(0.28-0.34): eso apunta a que el papel de NO/FB/LAL no es solo "más recurrencia" (ya
descartado aquí) sino algo más específico de esas conexiones, que ni las ganancias por
tipo entrenables logran compensar sobre el grafo restringido con este régimen.

## 2026-09-28 (6) — Rediseño de la tarea (punto C del revisor): implementado, cierra el hueco teórico, pero nada converge con 600 épocas

**Qué se hizo:** `task.py` gana `velocity_process="ou"` (proceso Ornstein-Uhlenbeck con
tiempo de correlación `tau_v`, en vez de ruido blanco independiente por paso) y
`model.py` gana `state_noise_std` (ruido de proceso aditivo a la dinámica recurrente,
tras la actualización). `train.py` y `task.py` pasan ambos parámetros de punta a punta
(comportamiento anterior exacto si se dejan por defecto). `task_v2_reference.py`
(nuevo) calcula las referencias de la tarea nueva.

**Por qué:** el revisor señaló 4 problemas de la tarea original: velocidad sin
correlación temporal (cambia casi de signo cada paso), ningún ensayo exige recorrer el
anillo completo (mediana ~60°), la dinámica no tiene ruido, y por eso un integrador
imperfecto (ganancia 0.54) pasa el criterio de "integra" con pérdida 0.138.

**Resultado / número clave — la referencia SÍ cierra el hueco:** con `T=300`,
`tau_v=30`, `max_av=0.15`, `hold_prob=0.3` (mediana 1.68 vueltas por ensayo, 63%
superan una vuelta completa), un integrador de ganancia 0.54 da pérdida 0.85 (memoria
sin integrar: 0.98; decodificador constante: 1.12) — ya no se cuela. Hace falta una
ganancia dentro de ±5% de 1.0 para pérdida < 0.2 (antes, ganancia 0.54 ya bastaba).
`results/task_v2_reference.json`.

**Pero la validación entrenada (3 semillas reales + 4 barajados, `launch_taskv2.sh`,
mismos hiperparámetros dinámicos y 600 épocas de siempre) no convergió NINGUNA:**
held-out 0.83-1.06 (todas cerca o peor que la memoria sin integrar, 0.98), pendiente
≈0 en las 7. La tarea nueva es más difícil de lo que el presupuesto de entrenamiento
actual puede resolver — ni siquiera la real se distingue de los barajados. Sin
converger nada, no se puede leer esto como confirmación ni como refutación de que el
rediseño cambie qué patrones integran.

**Siguiente paso (decisión con el usuario: más épocas, no currículo):** relanzadas
las mismas 7 corridas con 1400 épocas en vez de 600 (`launch_taskv2b.sh`,
`patience=200`), en curso al escribir esto.

## 2026-09-29 — Con 1400 épocas tampoco converge nada: no es presupuesto, es la receta

**Qué se hizo:** terminaron las 7 corridas de `launch_taskv2b.sh` (1400 épocas,
`patience=200`, mismos hiperparámetros dinámicos que siempre).

**Resultado / número clave:** prácticamente igual que con 600 épocas: held-out
0.82-1.04, pendiente entre −0.05 y +0.08 en las 7 (incluida la real). **Más del
doble de épocas no movió la aguja.** Descarta que fuera solo un problema de
presupuesto de entrenamiento; algo en la receta de hiperparámetros (`recurrent_gain`,
`tau`, `in_gain`, `cue_gain`, todos heredados de la tarea vieja, sin volver a
ajustar) o en la dificultad de optimización de la tarea nueva (BPTT sobre T=300 con
velocidad correlada, quizás un paisaje mucho más plano o con más mesetas que la tarea
original) impide que el descenso de gradiente encuentre la solución que sí existe con
los signos reales (la referencia de `task_v2_reference.json` confirma que una
ganancia cercana a 1.0 da pérdida ~0, aunque ese integrador ideal (cumsum) no es una red de esta clase de modelos: no demuestra que la solución exista en el espacio de
búsqueda; ver 2026-09-29 (3)).

**No se lanzó una tercera ronda sin confirmar con el usuario** (el patrón de la
sesión: cada ronda de cómputo grande se decide con él, no por iniciativa propia). El
rediseño de tarea queda como lo que es: cierra el hueco teórico (`task_v2_reference.json`),
pero no se ha podido entrenar nada sobre él con dos presupuestos de cómputo distintos;
haría falta currículo (empezar con parámetros más parecidos a la tarea vieja e ir
subiendo `tau_v`/T durante el entrenamiento) o repetir la búsqueda de régimen para
esta tarea, ninguno de los dos intentado todavía.

**Siguiente paso:** ninguno lanzado. Pendiente de decisión: ¿currículo, búsqueda de
régimen para la tarea nueva, o dejarlo declarado como limitación abierta?

## 2026-09-29 (2) — Currículo probado: tampoco converge. Tres intentos, mismo negativo

**Qué se hizo:** `train.py` gana `curriculum_epochs` / `curriculum_T_start` /
`curriculum_tau_v_start`: durante las primeras `curriculum_epochs` interpola
linealmente `T` y `tau_v` desde un punto de partida fácil hasta el objetivo; el
held-out siempre se evalúa en el objetivo. `launch_taskv2c.sh`: 1400 épocas, primeras
800 interpolando T 200->300 y tau_v 5->30 (T=200/tau_v=5 es casi tan fácil como la
tarea vieja: 0.52 vueltas medianas en vez de 1.57), las 600 restantes en el objetivo
fijo. Mismas 3 semillas reales + 4 barajados que las dos rondas anteriores.

**Por qué:** ni 600 ni 1400 épocas con T=300/tau_v=30 fijos desde el inicio
convergieron (entrada anterior); un currículo que empieza por la versión fácil de la
tarea es la vía estándar para evitar quedarse atascado en un óptimo local malo desde
el arranque.

**Resultado / número clave: tampoco converge nada.** held-out 0.93-1.07, pendiente
entre −0.04 y +0.02 en las 7 (incluida la real) — el mismo rango que sin currículo.
El progreso vigilado en vivo mostró `best_loss` estancado desde la época 0 hasta al
menos la 175 (22% del currículo) en la semilla 0, sin mejora visible después.

**Van tres intentos con el mismo resultado (600 épocas fijas, 1400 épocas fijas,
currículo de 800+600): ninguno consigue que el descenso de gradiente encuentre la
solución que sabemos que existe** (la referencia de ganancia ≈1.0 da pérdida ≈0,
`task_v2_reference.json`). Esto ya no parece un problema de presupuesto ni de
programa de entrenamiento, sino de la receta de hiperparámetros dinámicos
(`recurrent_gain=2`, `tau=10`, `in_gain=10`, `cue_gain=10`, heredados sin ajustar de
la tarea vieja) o de algo más estructural en cómo BPTT sobre T=300 con velocidad
correlada interactúa con esta red. No se ha intentado todavía: otra tasa de
aprendizaje, otro `recurrent_gain`/`tau` para esta tarea específicamente (equivalente
a la búsqueda de régimen que sí funcionó para diagnosticar el grafo PB+EB), u otra
inicialización.

**Se para aquí esta línea de trabajo sin lanzar una cuarta ronda de cómputo por
iniciativa propia.** El rediseño de tarea queda documentado como limitación abierta,
con el hueco teórico cerrado (`task_v2_reference.json`) pero sin ningún entrenamiento
que lo confirme empíricamente, tras tres intentos razonables.

**Siguiente paso:** decisión pendiente con el usuario. Opciones: (a) búsqueda de
régimen (sin entrenar, barata, igual que se hizo para PB+EB) específica para esta
tarea antes de reintentar el entrenamiento; (b) dejarlo declarado como limitación y
pasar el paquete al revisor tal como está.

## 2026-09-29 (3) — Tercera revisión externa: diagnóstico por regiones, tarea OU corregida y matices de simetría

**Qué se hizo:** (a) `region_analysis.py`: desfase PEN sin filtrar, peso por región/par de tipos y
ablación por región sobre los 3 modelos reales entrenados; `make_region_graphs.py` +
`launch_regions.sh`: reentrenar la real (3 semillas) sobre grafo completo sin NO y sobre PB+EB+NO.
(b) `task.py`: `increment_loss`; `model.py`/`task.py`: ruido de estado con semilla propia en held-out;
`task_v3_reference.py`, `transfer_check_v3.py`, `launch_taskv3.sh`. (c) `sign_symmetry_analysis.pair_strength`.
(d) Correcciones de texto (recuento 388/604, «salvedad cerrada», regla Delta7×EPG, explorer, figuras 5 y 6,
expert-review marcado histórico).

**Resultados clave:** desfase sin filtrar ≈6° (la dilución es real; la premisa del revisor de que EPG→PEN
está casi todo en PB era falsa, y el cuaderno también: solo 12 % está en PB). Ablación del NO: 0/3 integran
(held-out 0.43, pendiente 0.10); FB/LAL/CentralBrain: sin efecto. Reentrenando: sin NO 0/3 (0.30-0.36),
PB+EB+NO 3/3 (0.10-0.13). El NO solo tiene sinapsis PEN↔PEN. Tarea OU: velocidad filtrada p90 0.19 (anterior)
vs 0.047 (antigua) vs 0.055 (nueva); con la nueva y pérdida incremental, 3 reales convergen (0.12, pendiente
0.73-0.75) y 4 barajados no (0.35-0.46). Especificidad en PB+EB: real 0.31 vs 0.43-0.60. Simetría: pares casi
nulos y PEG/EPGt de salida débil; cambiar solo su signo mueve el held-out una mediana de 0.055.

**Limitaciones:** ablación sin reentrenar y reentrenamiento con hiperparámetros sin reajustar; 4 barajados
por neurona en la tarea nueva (no se repitió la enumeración de 64 patrones); no se ha probado por qué la
recurrencia PEN↔PEN es necesaria.

**Siguiente paso:** repetir la enumeración de 64 patrones sobre la tarea OU corregida; el revisor puede
repetir los cálculos con `results/region_analysis.json`.

## 2026-10-09 — Enumeración de los 64 patrones sobre la tarea OU corregida (DISEÑO, fijado antes de ver resultados)

**Qué se hizo:** `data/interim/launch_types_v3.sh`: los 63 patrones de signo por tipo distintos de la
real (`control_type_mask` 0..63 salvo 1) × semillas 0, 1, 2 = 189 corridas (`taskv3_mask{m}_s{s}_p`),
con la receta exacta de `launch_taskv3.sh` (800 épocas, `T=500`, `tau_v=50`, `max_av=0.04`,
`state_noise_std=0.02`, `loss_type='increment'`, `checkpoint_selection='ema'`, mismos hiperparámetros
dinámicos). La máscara 1 se reutiliza de `taskv3_real_seed{0,1,2}_p` (prueba de humo de 3 épocas con
`control_type_mask=1`: 0 % de aristas cambiadas respecto a la real). Orden por semilla (primero las 63
de la semilla 0), 14 en paralelo, reanudable.

**Por qué:** era la limitación declarada de la tarea corregida (solo 4 barajados por neurona); el
resultado central del informe (la real no es especial entre los 64 patrones) solo estaba medido en la
tarea antigua, que la 3ª revisión cuestionó.

**Criterio primario (fijado antes):** análogo directo del de la entrada 2026-09-20. Métrica: pérdida
incremental en held-out (la que se entrena). «Integra» = pérdida incremental < 0.538 («recordar
theta0» en esta tarea, `task_v3_reference.json`) Y pendiente > 0.5; por patrón, voto de mayoría de
3 semillas. p exacto = puesto de la máscara 1 por pérdida incremental media / 64. Descriptivos: nº de
patrones que integran, fracción por signo de cada tipo, y concordancia con el veredicto de la tarea
antigua (`types_analysis_3seeds.json`). No se excluye ninguna corrida. Si el cómputo se corta antes
de las 3 semillas, se informa con las semillas completas que haya, diciéndolo.

**Riesgos declarados:** los mismos que en 2026-09-20 (hiperparámetros elegidos con la real, signo por
tipo de origen, `ring_sign=+1`); además el umbral de pérdida es laxo (los barajados por neurona dieron
0.35-0.46) y es la pendiente la que discrimina.

**Siguiente paso:** analizar con `analyze_types_v3.py` cuando terminen.

## 2026-10-09 (2) — 64 patrones sobre la tarea OU corregida, semilla 0: la real baja al puesto 37 de 64

**Qué se hizo:** terminaron las 63 corridas de la semilla 0 (ninguna con error); `analyze_types_v3.py`
-> `results/types_analysis_v3.json` con el criterio de la entrada anterior. Semillas 1 y 2 en curso.

**Resultado / número clave (1 semilla, provisional):** **41/64 patrones integran** (pérdida incremental
< 0.538 y pendiente > 0.5). La real (máscara 1): 0.115, pendiente 0.74, **puesto 37 de 64
(p = 37/64 = 0.578)**. Mejores: máscaras 25, 49, 45, 61, 37, 15, 33, 26 (0.029-0.040, pendiente
0.94-1.03); la 26 (EPG+PEG+PEN_a inhiben, Delta7 excita) está entre ellas. Por tipo (integran con el
tipo excitador / inhibidor, de 32): D7 11/30; EPG 23/18; EPGt 19/22; PEG 20/21; PEN_a 15/26; PEN_b
20/21. Concordancia con la tarea antigua (mayoría de 3 semillas): 24 integran en ambas, 4 solo en la
antigua, 17 solo en la nueva, 19 en ninguna.

**Lectura:** la conclusión del informe se mantiene y se refuerza en la tarea corregida (la real integra
pero no es especial; 36 patrones la superan). Cambia el descriptivo por tipo: el contraste de PEN_b
excitador de la tarea antigua (20/32 vs 6/32) desaparece (20 vs 21) y Delta7 inhibidor pasa a ser casi
suficiente (30/32) sin ser necesario (11/32 integran con Delta7 excitador). Varios patrones alcanzan
pendiente ≈1 frente a 0.74 de la real: la real es un integrador más compresivo que muchas alternativas.

**Limitaciones:** una semilla (en la tarea antigua 16/63 patrones no fueron unánimes entre semillas);
el umbral de pérdida es laxo y discrimina la pendiente; descriptivos por tipo sin test.

**Siguiente paso:** rehacer el análisis con las 3 semillas y llevar las cifras al borrador, al resumen
en inglés y al README.

## 2026-10-12 — 64 patrones sobre la tarea OU corregida, 3 semillas: la real, puesto 30 de 64

**Qué se hizo:** terminaron las 189 corridas (semillas 0, 1, 2; ninguna con error en los logs);
`analyze_types_v3.py` -> `results/types_analysis_v3.json` con el criterio fijado el 2026-10-09 (sin cambios).
La semilla 2 se había cortado a 28/63 y se completó relanzando `launch_types_v3.sh` (reanudable).

**Resultado / número clave:** **37/64 patrones integran por mayoría** (32/64 en las 3 semillas). La real
(máscara 1): pérdida incremental media 0.121, pendiente 0.74, integra en 3/3, **puesto 30 de 64
(p = 30/64 = 0.469)**. Mejores: máscaras 25, 49, 45, 41, 53, 37, 15, 61 (0.032-0.044, pendiente
0.93-0.99), todas con Delta7 inhibidor. Por tipo (integran con el tipo excitador / inhibidor, de 32):
D7 7/30; EPG 20/17; EPGt 19/18; PEG 18/19; PEN_a 15/22; PEN_b 18/19. Concordancia con la tarea antigua
(mayoría de 3 semillas): 23 en ambas, 5 solo en la antigua, 14 solo en la nueva, 22 en ninguna.

**Lectura:** se mantiene la conclusión del informe: la real integra pero no es especial (29 patrones la
superan). Respecto a la semilla 0 sola (41 integran, puesto 37) las cifras se mueven, como se advirtió.
Descriptivo por tipo: solo Delta7 separa (inhibidor 30/32 vs excitador 7/32); el contraste de PEN_b de la
tarea antigua (20/32 vs 6/32) desaparece (18 vs 19). Delta7 inhibidor es casi suficiente, no necesario.
La pendiente de los mejores (≈0.95) supera a la de la real (0.74): la real es más compresiva.

**Limitaciones:** 3 semillas; umbral de pérdida laxo (discrimina la pendiente); descriptivos por tipo sin
test; el nº que integra depende del criterio de agregación (mayoría 37, unánime 32).

**Siguiente paso:** llevar estas cifras al borrador (`docs/preprint-discussion.md`: ítem «Trabajo futuro
(pendiente)» de la línea ~737), a `docs/summary-en.md` y al README.

## 2026-10-12 (2) — Robustez de la enumeración corregida (post hoc) y revisión final del repositorio

**Qué se hizo:** análisis exploratorio, posterior a ver los resultados, añadido a `analyze_types_v3.py`
(bloque `exploratory_post_hoc` de `results/types_analysis_v3.json`); `collect_results.py` reconstruyó
`runs_summary.csv` (+189 filas de la enumeración); los lanzadores `launch_*.sh` (hasta ahora solo en
`data/interim/`, fuera del repositorio) se versionaron en `scripts/` con rutas relativas.

**Resultado / número clave:** el puesto de la real es 37, 36 y 37 por semilla individual (30 por la media
de 3). Recuento de patrones que integran según la pendiente exigida: 37 (> 0.5 y > 0.7), 32 (> 0.8), 20
(> 0.9); con > 0.8 la real (pendiente 0.74) no integra en ninguna semilla; el umbral de pérdida no cambia
el recuento. Condicionado a Delta7 inhibidor (32 patrones): 30 integran y la real es el puesto 28 de 32; con
Delta7 excitador integran 7 de 32 y el mejor llega a 0.058.

**Lectura:** «la real no es especial» es robusta a semilla y umbrales; «la real integra» depende del
umbral de pendiente. La tarea corregida selecciona el bit conocido (Delta7 inhibe) y nada más específico:
fijado ese bit, los otros cinco casi no importan. El «p = puesto/64» es descriptivo, no un contraste.

**Limitaciones:** análisis post hoc (no estaba en el criterio fijado el 2026-10-09); 3 semillas.

**Siguiente paso (huecos para cerrar la lógica del estudio, en orden):** (1) repetir el aprendiz de signos
libres (por arista y por 6 signos por tipo) sobre la tarea corregida: es lo que falta para responder la
pregunta original; (2) 5-10 semillas y nula explícita; (3) hiperparámetros sin ayuda de la real;
(4) variar el modelo; (5) verificar la bibliografía del posicionamiento.

## 2026-10-12 (3) — Aprendiz de signos sobre la tarea corregida (DISEÑO Y CRITERIO, fijados antes de ver resultados)

**Qué se hace:** el experimento 1 del trabajo futuro: ¿recupera un aprendiz de signos la química real cuando la tarea sí
converge? Receta idéntica a `launch_types_v3.sh` (800 épocas, `T=500`, `tau_v=50`, `max_av=0.04`, `state_noise_std=0.02`,
`loss_type='increment'`, `checkpoint_selection='ema'`, `sign_reg=0`, ganancias/sesgos por tipo entrenables); solo cambia
quién decide el signo. Código: `type_sign_learner` / `type_sign_init` en `train.py` y `model.py` (6 parámetros
`tanh(type_sign_param)`, uno por tipo de origen, la misma parametrización que la enumeración de 64 patrones);
`scripts/launch_signlearner.sh`; análisis en `src/cx_net/analyze_sign_learner.py` -> `results/sign_learner_v3.json`.
Comprobado: con el defecto, `loss_first` de una corrida de la enumeración se reproduce exactamente.

**Brazos (32 corridas, sin exclusiones):**
- **A (primario):** 6 signos por tipo, inicio neutro N(0, 0.1), semillas 0-9.
- **B (primario):** 6 signos por tipo, inicio binario ±1 aleatorio (un inicio distinto por semilla), semillas 0-9.
- **C (descriptivo):** un signo por arista (9.160), inicio neutro, semillas 0-5.
- **D (control de estabilidad):** 6 signos por tipo, inicio en la asignación real, semillas 0-5.

**Medidas por corrida:** patrón aprendido = tipos cuyo signo final es negativo (máscara con los mismos bits que la
enumeración); «integra» = pérdida incremental held-out < 0.538 y pendiente > 0.5, evaluado con el modelo continuo Y con los
signos binarizados a ±1; acuerdo entre el signo aprendido por arista y el neurotransmisor anotado.

**Criterio de lectura (fijado antes), sobre los brazos A y B por separado:**
1. **La química real es recuperada:** el patrón aprendido es la máscara 1 (solo Delta7 inhibe) en ≥ 8 de 10 semillas y
   esas corridas integran. Referencia de azar: 1/37 (elegir al azar entre los 37 patrones que integran en la enumeración);
   se informa la cola binomial P(X ≥ k).
2. **El signo no es identificable con esta tarea:** ≥ 7 de 10 corridas integran, pero la máscara 1 aparece en ≤ 2 de 10
   y hay varias máscaras distintas. Se informa además cuántas tienen Delta7 inhibidor y cuántas son patrones que integran
   en la enumeración.
3. **Fallo de optimización (no informativo sobre identificabilidad):** < 5 de 10 corridas integran. Se declara como
   límite del aprendiz, no como resultado sobre la química.
4. Cualquier otro resultado se describe sin etiqueta, con las cifras.
- **D:** fracción de semillas que terminan en la máscara 1 y siguen integrando (¿es la real un atractor estable?).
- **C:** solo descriptivo (acuerdo con el signo real, fracción inhibidora por tipo de origen); sin umbrales.
- Una corrida que falle por error técnico se repite con la misma semilla; las que no convergen cuentan como «no integra».

**Riesgos declarados:** hiperparámetros dinámicos elegidos con la real; la parametrización `tanh` con inicio cercano a
cero puede dar mesetas (el resultado más probable es el caso 3 o 4); un solo `ring_sign` (+1); 10 semillas.

**Siguiente paso:** lanzar (`bash scripts/launch_signlearner.sh`, 16 procesos en paralelo, 1 hilo cada uno; solo CPU:
el PyTorch instalado es CPU y la carga son 500 pasos secuenciales sobre 152 neuronas, que no se beneficia de GPU) y
analizar con `analyze_sign_learner.py`.

## 2026-10-12 (4) — Aprendiz de signos sobre la tarea corregida: RESULTADO (32 corridas, ninguna excluida)

Datos: `results/sign_learner_v3.json` (`analyze_sign_learner.py`). Aplicado el criterio de la entrada (3), sin cambios.

| brazo | integran (continuo / binarizado ±1) | patrón == real | Delta7 inhibidor | patrones distintos | acuerdo con el signo real por arista |
|---|---|---|---|---|---|
| A (6 signos por tipo, inicio neutro) | 7/10 / 3/10 | **0/10** | 6/10 | 9 en 10 corridas | 0.68 |
| B (6 signos por tipo, inicio binario aleatorio) | 6/10 / 6/10 | **0/10** | 3/10 | 9 en 10 corridas | 0.46 |
| C (un signo por arista, inicio neutro) | 4/6 / 4/6 | n/a | n/a | n/a | 0.51 (≈ azar) |
| D (6 signos por tipo, inicio en la real) | 6/6 / 6/6 | **0/6** | 6/6 | 2 (máscara 13 en 5, máscara 9 en 1) | 0.92 |

**Lectura con el criterio preregistrado:**
- La química real **no se recupera** en ninguna de las 20 corridas A+B (0/20; cola binomial con azar 1/37: P(X ≥ 0) = 1,
  es decir, tampoco hay un solo acierto). Ningún patrón domina: 9 máscaras distintas en 10 corridas en cada brazo.
- **Etiqueta formal, con una ambigüedad que declaro:** el criterio dice que «integra» se evalúa con el modelo continuo Y con
  los signos binarizados, sin decir si hacen falta ambos. Con el continuo, A cumple el caso 2 (7/10 integran, real 0/10, varias
  máscaras: «no identificable»). Con ambos a la vez, A tiene 3/10 y cae en el caso 3 (fallo de optimización). B (6/10 en ambos
  modos) no entra en ningún caso: queda sin etiqueta. No elijo la lectura que más convenga: la conclusión sólida es solo
  «0/20 recuperan la real», que no depende de la etiqueta.
- **No se puede concluir** que el gradiente no *pueda* encontrar la real, sino que, con esta receta y esta tarea, no la prefiere:
  coherente con que 37 de 64 patrones integran y la real es el puesto 30 (el aprendiz cae en uno de muchos óptimos equivalentes).
- **D (control de estabilidad):** arrancando en la real, las 6 corridas siguen integrando pero **ninguna conserva la máscara 1**:
  5 derivan a 13 (Delta7 + EPGt + PEG inhiben) y 1 a 9 (Delta7 + PEG). Delta7 sigue siendo inhibidor en 6/6 (el único bit que
  conocemos). La real no es un atractor del entrenamiento: añadir inhibidores en otros tipos no empeora la tarea.
- **C:** acuerdo con el signo real ≈ 0.51, indistinguible de asignar signos al azar (el 0.92 de D es casi todo el inicio).
- Tres corridas de A (semillas 4, 5, 7) y cuatro de B (1, 2, 7, 9) no integran (pérdida incremental 0.38-0.57, pendiente ≈0-0.2:
  el óptimo local «memoria sin integrar»); se cuentan como «no integra» (criterio fijado). Ninguna se excluyó.

**Límites:** hiperparámetros dinámicos elegidos con la real; un solo `ring_sign`; 10 semillas por brazo; potencia no calculada;
«integra» es un umbral binario arbitrario.

**Siguiente paso:** actualizar informe, resumen, README, web y post; las prioridades 2-5 del trabajo futuro siguen igual.

## 2026-10-12 (5) — Cierre del estudio

**Decisión:** se da por concluido el estudio como informe metodológico y negativo. No se lanza el experimento con una tarea que
discrimine químicas (prioridad 3-4 del trabajo futuro): requiere diseñar una tarea nueva, con resultado incierto, y el
hallazgo central (la tarea no selecciona la química; el aprendiz no la prefiere) ya está acotado y documentado.
**Pendiente solo si se retoma o se quiere publicar:** verificar la bibliografía (referencias no leídas en texto completo),
más réplicas con nula explícita y hiperparámetros ciegos a la real. Post de LinkedIn actualizado con el 0/20.

## Plantilla para próximas entradas

```
## AAAA-MM-DD — <título breve>

**Qué se hizo:**
**Por qué:**
**Resultado / número clave:**
**Siguiente paso:**
```

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

## Plantilla para próximas entradas

```
## AAAA-MM-DD — <título breve>

**Qué se hizo:**
**Por qué:**
**Resultado / número clave:**
**Siguiente paso:**
```

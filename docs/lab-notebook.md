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

## Plantilla para próximas entradas

```
## AAAA-MM-DD — <título breve>

**Qué se hizo:**
**Por qué:**
**Resultado / número clave:**
**Siguiente paso:**
```

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

## Plantilla para próximas entradas

```
## AAAA-MM-DD — <título breve>

**Qué se hizo:**
**Por qué:**
**Resultado / número clave:**
**Siguiente paso:**
```

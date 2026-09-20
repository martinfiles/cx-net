# Revisión crítica interna (2026-09-20)

Lectura adversarial del borrador (`docs/preprint-discussion.md`, v2) desde el punto de vista de
un revisor de neurociencia computacional del sistema de dirección de cabeza y del modelado
con restricción de conectoma. Es una revisión propia, no independiente: la hizo quien construyó
el análisis, con el sesgo que eso implica. **Lo ideal es que la repita alguien del campo.**

## Qué afirma el borrador

Un aprendiz de signos, restringido solo por la topología del núcleo del sistema de dirección de
cabeza de *Drosophila*, no recupera el neurotransmisor anotado; el diseño original no lo permitía
(tarea trivial, sin ancla, mapeo angular equivocado); con ganancias por tipo existe una solución
que integra el rumbo, pero no es específica de la asignación real (26/64 patrones por tipo; la
real, puesto 22); y un aprendiz genérico no la encuentra.

## Preocupaciones principales

**M1. La hipótesis es casi trivial para este circuito.** El neurotransmisor anotado es función
exacta del tipo (Delta7 = glutamato; el resto, acetilcolina). «Recuperar el neurotransmisor»
equivale a recuperar un bit que la anatomía funcional ya establece (Delta7 inhibe). Incluso un
resultado positivo habría informado poco. *Atendido:* el borrador lo dice de forma explícita
(sección 2) y reformula la pregunta de fondo (¿restringen función y topología el signo más allá
de lo conocido?).

**M2. El «ground truth» es una predicción y hay supuestos de signo no discutidos.** `predictedNt`
sale de un clasificador sobre EM; se asume ley de Dale y ACh→excita, glutamato→inhibe, sin
co-transmisión ni dependencia del receptor (receptores muscarínicos inhibidores, receptores de
glutamato excitadores). *Atendido:* terminología corregida y limitación añadida. *Abierto:* citar
las referencias concretas que respaldan cada tipo.

**M3. «Tasas negativas no fisiológicas» era demasiado fuerte.** Muchos modelos de anillo usan
tasas como desviaciones de una basal; una neurona inhibidora por debajo de su basal desinhibe.
El problema real es la **falta de identificabilidad del signo** en un modelo con basal nula, no
que el resultado sea «no biológico». *Atendido:* reformulado (sección 6, retractaciones, README,
figura 5). *Abierto:* la única prueba con basal explícita (sigmoide) no fue concluyente.

**M4. La clase de modelo está poco restringida frente al circuito real y la solución no está
caracterizada mecanísticamente.** Magnitudes = conteo de sinapsis normalizado por destino, una
sola constante de tiempo, entrada de velocidad sintética, sin ring neurons. Un revisor pediría
(a) perfil del bump y su anchura, (b) ganancia velocidad→desplazamiento, (c) comparación con los
modelos de anillo basados en conectoma ajustados a mano. *Atendido en parte:* se midió (b) y la
localización del bump: las redes que «integran» responden de forma antisimétrica y aproximadamente
proporcional (ganancia 0.54-0.76, R² 0.84-0.93), pero con saturación; con velocidades constantes
grandes ningún modelo desplaza el bump; la localización del bump es moderada (0.62 en la solución
real; 0.34-0.37 en otras). *Abierto:* (a) y (c).

**M5. Estadística exploratoria.** Una corrida por patrón; «integra» es un umbral binario
arbitrario; las fracciones por tipo son marginales de patrones no independientes y se leen como
efectos causales por error; sin incertidumbre por corrida; las diferencias entre puestos vecinos
están dentro del ruido de optimización (réplicas de la real: 0.106-0.124). *Atendido:* declarado
como exploratorio y limitaciones ampliadas. *A favor del borrador:* los hiperparámetros se
eligieron con la asignación real, lo que la favorece; que no destaque refuerza el nulo.
*Abierto:* replicar cada patrón con varias semillas.

**M6. Resolución mínima del test de H1 por tipo.** Con 6 tipos hay 64 patrones, así que el
p más pequeño por coincidencia exacta es 1/64. Cualquier test de «el aprendiz recupera el
neurotransmisor» a nivel de tipo tiene esa resolución. *Atendido:* añadido a la sección 3.

**M7. La geometría del anillo no es nueva.** El espejo entre hemisferios y el desfase L/R están
descritos (Wolff et al., 2015; Hulse et al., 2021). La aportación es un procedimiento reproducible
en MaleCNS y la corrección de los mapeos previos. El ajuste por PCA supone un EB casi plano y
muestreo angular parejo. *Atendido:* reformulado. *Abierto:* validarlo en hemibrain frente a las
tablas publicadas y contra el recuento de 16 cuñas.

**M8. Falta trabajo previo.** No hay sección de posicionamiento (modelos de anillo basados en
conectoma, Lappalainen et al. 2024, Dhiman 2026, Eckstein et al. 2024, Berg et al. 2026).
*Atendido en parte:* subsección añadida, con referencias marcadas como por verificar. *Abierto:*
revisión sistemática y comparación directa.

## Preocupaciones menores

- La tabla de fracciones por tipo (figura 4) sugiere efectos que no lo son: el título usaba
  «pesan más»; conviene «se asocian con».
- «Ground truth» → «neurotransmisor anotado» en todo el texto (hecho en el borrador; faltan
  comentarios en el código).
- El criterio «integra» debería definirse en el resumen.
- La comparación de la fig. 1 mezcla dos conjuntos de referencia (tarea original); indicar que
  son de la tarea sin ancla.
- La afirmación «los 12 barajados fallan porque rompen la coherencia por tipo» es una explicación
  plausible, no probada; los patrones por tipo la respaldan indirectamente.

## Veredicto

Como **nota metodológica y negativa** es defendible tras las correcciones aplicadas: sus cifras son
trazables (`results/`), las retractaciones están explícitas y no afirma nada sobre la biología del
sistema. **No** es defendible como resultado sobre la química real de la mosca, y no debería
presentarse así. Antes de una publicación formal haría falta al menos: (1) replicar los patrones
con varias semillas, (2) caracterizar el bump y compararlo con modelos ajustados a mano, (3) validar
el mapeo de fase en hemibrain, (4) una revisión de la literatura por alguien del campo.

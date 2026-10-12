# CX-Net

Validación cruzada entre pesos aprendidos por una red neuronal restringida por el connectoma del complejo central de *Drosophila melanogaster* y su neurotransmisor real anotado.

## Pregunta de investigación

Si se entrena una red neuronal cuya única restricción es la topología sináptica real del complejo central (sin indicarle qué conexiones son excitadoras o inhibidoras), dejando el signo de cada peso completamente libre, ¿converge por sí sola hacia el neurotransmisor real ya anotado en el dataset, más allá de lo esperable por azar?

## Estado

Experimentos cerrados el 2026-10-12; informe metodológico y negativo en redacción
(exploratorio, sin revisión por pares).

Resultado principal: **no hay evidencia de que un aprendiz de signos recupere el
neurotransmisor real, y el diseño no permitía encontrarla.** Hallazgos: (1) la
tarea original era resoluble de forma trivial (un decodificador constante
superaba a los modelos entrenados); (2) la fase angular real de cada neurona en
el EB, medida con coordenadas de sinapsis, es un espejo entre hemisferios que
los mapeos asumidos no recogían; (3) con ganancias por tipo celular existe una
solución que integra el rumbo, pero no es específica de la química real (28 de
64 patrones de signo por tipo la alcanzan, con 3 semillas; la real, puesto 14; con la tarea de
velocidad corregida, 37 de 64 y puesto 30) y usa tasas con
signo (desviaciones de una basal nula) que hacen el signo poco identificable; (4) un aprendiz de signos genérico no la
encuentra; (5) en este subcircuito el neurotransmisor es función exacta del tipo
celular (Delta7 = glutamato, resto = acetilcolina). Detalle y retractaciones en
[`docs/preprint-discussion.md`](docs/preprint-discussion.md) y
[`docs/lab-notebook.md`](docs/lab-notebook.md).
Ver hoja de ruta más abajo.

## Figuras

Se regeneran con `python -m src.cx_net.make_figures [--lang es|en|all]` (leen solo de `results/`).
Versiones en inglés en [`docs/figures/en/`](docs/figures/en/); resumen en inglés en
[`docs/summary-en.md`](docs/summary-en.md) (English summary). Explorador interactivo de los 64
patrones de signo: [`docs/explorer.html`](docs/explorer.html) (abrir localmente en el navegador).

![Un decodificador inerte supera a todos los modelos entrenados](docs/figures/fig1.png)
![Los dos hemisferios recorren el anillo en sentidos opuestos](docs/figures/fig2.png)
![28 de 64 patrones integran; la asignación real es uno más](docs/figures/fig3.png)
![Ningún signo por tipo basta](docs/figures/fig4.png)
![La solución con la química real mantiene Delta7 por debajo de la basal](docs/figures/fig5.png)
![Las redes que integran mueven un bump ancho, con ganancia sigmoidal](docs/figures/fig6.png)
![Tarea corregida: 37 de 64 integran; la real, puesto 30](docs/figures/fig7.png)

Las figuras 3 y 4 corresponden a la primera tarea anclada (28 de 64, puesto 14); la figura 7, a la tarea
OU corregida (3ª revisión). Las dos llevan a la misma conclusión.

## Datos

- Conectividad del subcircuito CX (PB, EB, FB, NO) vía [neuprint-python](https://github.com/connectome-neuprint/neuprint-python), tanto para el piloto en hemibrain como para `male-cns:v1.0` (ambos expuestos por neuprint.janelia.org).
- Neurotransmisor real anotado por síntesis (Eckstein et al., 2024) — reservado como *ground truth* oculto, nunca visto durante el entrenamiento.

## Fases

| Fase | Descripción | Estado |
|---|---|---|
| 1 | Extracción del grafo (sin signo) | hecha |
| 2 | Modelo de signo libre (PyTorch) | hecha |
| 3 | Controles (nula, referencia trivial, 64 patrones por tipo) | hecha |
| 4 | Evaluación y desglose por neuropilo/tipo celular | hecha |
| 5 | Redacción y difusión (informe, web, LinkedIn) | en curso |

Pendiente principal: repetir el aprendiz de signos libres sobre la tarea corregida (sección 11 del informe).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # añadir NEUPRINT_TOKEN (neuprint.janelia.org > Account > Auth Token)
```

## Uso

```bash
python -m src.cx_net.extract_graph
```

Genera `data/raw/graph_no_sign.csv` (topología pura, sin signo) y
`data/raw/ground_truth_nt.csv` (neurotransmisor real, uso restringido a la
fase de evaluación). Ambos quedan fuera del repositorio (`.gitignore`).

Fases posteriores (módulos en `src/cx_net/`; los resultados se escriben en
`data/interim/`, fuera del repositorio):

```bash
python -m src.cx_net.train          # entrena el modelo de signo libre
python -m src.cx_net.evaluate       # H1 (test de permutación) y H2 (por tipo celular)
python -m src.cx_net.extract_eb_angles  # fase angular real de cada neurona (sinapsis del EB, neuPrint)
python -m src.cx_net.analyze_types           # análisis de los 64 patrones (1 semilla)
python -m src.cx_net.analyze_types_3seeds    # agregado de los 64 patrones con 3 semillas
python -m src.cx_net.analyze_types_v3        # los 64 patrones sobre la tarea OU corregida (3 semillas)
```

Los barridos multi-semilla y las variantes (`hold_prob`, `perturb_amp`,
`sign_reg`, tarea anclada, parámetros por tipo, controles de signos) se lanzaron
llamando a `train()` con distintos argumentos; ver el cuaderno para las
configuraciones exactas. **Entrenar siempre con el intérprete del `.venv`**: con
torch 2.4.1 (Python del sistema) el gradiente de `atan2(0,0)` es `nan`.

Los lanzadores de las corridas (`launch_*.sh`) están en [`scripts/`](scripts/) (se ejecutan desde cualquier
directorio; escriben en `data/interim/`, fuera del repositorio). La enumeración sobre la tarea corregida es
`scripts/launch_types_v3.sh` (189 corridas, reanudable; 14 procesos en paralelo, 1 hilo cada uno).

Registro de decisiones y resultados: [`docs/lab-notebook.md`](docs/lab-notebook.md).

## Reproducibilidad

- **Datos.** No se redistribuyen: `data/` está fuera del repositorio. Hace falta un
  token personal de neuPrint (`.env`) y `python -m src.cx_net.extract_graph` (por defecto `--dataset male-cns:v1.0`) para
  regenerar el grafo de `male-cns:v1.0`; `python -m src.cx_net.extract_eb_angles` regenera
  la fase angular de cada neurona; `python -m src.cx_net.pen_offset_by_region` (también
  necesita el token) regenera el desfase de fase de los PEN separado por compartimento
  (PB de entrada, EB de salida; sección 5).
- **Resultados.** `results/` contiene los resúmenes que respaldan el borrador
  (`runs_summary.csv`: una fila por corrida; análisis de los 64 patrones, del refuerzo
  y de la búsqueda de régimen; ángulos agregados por glomérulo). `python -m
  src.cx_net.collect_results` reconstruye `runs_summary.csv` y los JSON/CSV pequeños
  desde `data/interim/`; `python -m src.cx_net.analyze_types_3seeds` reconstruye
  `types_analysis_3seeds.json` y `types_explorer_3seeds.json` (la cifra principal,
  28/64 y puesto 14, sección 6) a partir de las mismas corridas; `python -m
  src.cx_net.analyze_types_v3` hace lo propio para la tarea corregida (37/64 y puesto 30,
  sección 11, con un bloque exploratorio posterior de robustez). Ninguno de los dos
  requiere reentrenar: leen los `results_*.json` ya guardados en `data/interim/`.
- **Entorno.** Versiones usadas: Python 3.11.9, torch 2.14.0, numpy 2.4.6, pandas 2.3.3,
  matplotlib 3.11.2, neuprint-python 0.6.3 (ver `requirements.txt`). Fija tu versión de PyTorch: con `torch` 2.4.1 el gradiente de `atan2(0,0)`
  era `nan` (corregido en `task.decode_heading`; el entrenamiento por defecto reproduce
  exactamente `epoch 0/25` de `signreg05_seed0` en `torch` 2.4.1 y 2.14).
- **Coste.** Cada entrenamiento de 600 épocas tarda ≈40-50 min en CPU con ~20 en
  paralelo; la enumeración de 64 patrones lleva ≈2,5 h en 16 núcleos.
- **Cuaderno.** [`docs/lab-notebook.md`](docs/lab-notebook.md) registra cada decisión,
  incluidos los errores y sus correcciones; las afirmaciones retractadas están en la
  tabla de la sección 9 de [`docs/preprint-discussion.md`](docs/preprint-discussion.md).

## Datos, licencia y citas

- **Código:** MIT (`LICENSE`).
- **Datos de conectoma:** MaleCNS (`male-cns:v1.0`) se distribuye bajo **CC-BY 4.0**;
  este repositorio no los redistribuye. Los ficheros derivados de `results/` (ángulos
  agregados por glomérulo, resúmenes de entrenamientos) se ofrecen con la misma atribución.
  Si usas este trabajo, cita el conectoma: Berg et al. (2026), *Sexual dimorphism in the
  complete Drosophila male central nervous system connectome*, Cell
  ([texto](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6)); preprint en bioRxiv,
  doi:10.1101/2025.10.09.680999. Neurotransmisor predicho: Eckstein et al. (2024).
- **Revisión crítica interna:** [`docs/expert-review.md`](docs/expert-review.md).

## Referencias clave

- Lappalainen, J.K. et al. (2024). *Connectome-constrained networks predict neural activity across the fly visual system.* Nature.
- Hulse, B.K. et al. (2021). *A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection.* eLife.
- Dhiman, S. (2026). *Topological Sensitivity in Connectome-Constrained Neural Networks.* arXiv:2604.04033.

## Stack

Python · PyTorch · pandas/NumPy · neuprint-python · matplotlib

## Autor

Martín Barros Iglesias

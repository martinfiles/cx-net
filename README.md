# CX-Net

Validación cruzada entre pesos aprendidos por una red neuronal restringida por el connectoma del complejo central de *Drosophila melanogaster* y su neurotransmisor real anotado.

## Pregunta de investigación

Si se entrena una red neuronal cuya única restricción es la topología sináptica real del complejo central (sin indicarle qué conexiones son excitadoras o inhibidoras), dejando el signo de cada peso completamente libre, ¿converge por sí sola hacia el neurotransmisor real ya anotado en el dataset, más allá de lo esperable por azar?

## Estado

Fase experimental cerrada (2026-09-18); en redacción del preprint.
Resultado principal (negativo): la red aprende la tarea de integración de
rumbo, pero el signo aprendido no coincide con el neurotransmisor real más
de lo esperable por azar, ni siquiera con regularización que fuerza la
polarización del signo (8 semillas). Un control positivo de potencia indica
que el test detectaría un acuerdo de ≈+3.3 pp o más sobre el azar con ≥80%
de probabilidad. Detalle en
[`docs/preprint-discussion.md`](docs/preprint-discussion.md) y
[`docs/lab-notebook.md`](docs/lab-notebook.md).
Ver hoja de ruta más abajo.

## Datos

- Conectividad del subcircuito CX (PB, EB, FB, NO) vía [neuprint-python](https://github.com/connectome-neuprint/neuprint-python) (hemibrain, dataset piloto) y [caveclient](https://github.com/CAVEconnectome/CAVEclient) (MaleCNS v1.0).
- Neurotransmisor real anotado por síntesis (Eckstein et al., 2024) — reservado como *ground truth* oculto, nunca visto durante el entrenamiento.

## Roadmap

| Fase | Descripción | Semanas |
|---|---|---|
| 1 | Extracción del grafo (sin signo) | 1–3 |
| 2 | Modelo de signo libre (PyTorch + PyTorch Geometric) | 3–6 |
| 3 | Controles estadísticos (modelo nulo, preserva grado) | 6–8 |
| 4 | Evaluación y desglose por neuropilo/tipo celular | 8–10 |
| 5 | Redacción y difusión (preprint + LinkedIn) | 10–12 |

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
python -m src.cx_net.power_control  # control positivo de potencia del test de H1
```

Los barridos multi-semilla y las variantes (`hold_prob`, `perturb_amp`,
`sign_reg`) se lanzaron llamando a `train()` con distintos argumentos; ver
el cuaderno para las configuraciones exactas.

Registro de decisiones y resultados: [`docs/lab-notebook.md`](docs/lab-notebook.md).

## Referencias clave

- Lappalainen, J.K. et al. (2024). *Connectome-constrained networks predict neural activity across the fly visual system.* Nature.
- Hulse, B.K. et al. (2021). *A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection.* eLife.
- Dhiman, S. (2026). *Topological Sensitivity in Connectome-Constrained Neural Networks.* arXiv:2604.04033.

## Stack

Python · PyTorch · PyTorch Geometric · neuprint-python · caveclient · navis · NetworkX

## Autor

Martín Barros Iglesias

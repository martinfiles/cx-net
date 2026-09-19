"""
Posición angular real de cada neurona de compás/conductora alrededor del anillo
del EB, medida a partir de las coordenadas 3D de sus sinapsis en neuPrint
(male-cns:v1.0). Sustituye la aproximación por glomérulo de `graph_utils`,
que contradecía la anatomía (ver lab-notebook, entradas 2026-09-19 (4) y (5)).

Método: se reúnen las sinapsis (pre y post, `primary_only`) en el ROI `EB` de
EPG, EPGt, PEN_a, PEN_b y PEG. El EB es un toro, así que la nube de sinapsis
es aproximadamente plana: se ajusta un plano por PCA (dos primeros componentes,
~89% de la varianza) y el ángulo de cada sinapsis es atan2 en ese plano
alrededor del centroide. El ángulo de una neurona es la media circular de sus
sinapsis ponderada por la confianza de la sinapsis. No se usa ninguna
información de neurotransmisor.

El signo del ángulo (sentido horario/antihorario) es arbitrario en este método;
`graph_utils.load_cx_graph(ring_sign=...)` lo fija después.

Salida: data/raw/malecns/eb_angles.csv (bodyId, eb_angle [rad], concentration
[0-1, longitud del vector medio: 1 = todas las sinapsis en la misma cuña],
n_syn). data/raw/ está fuera del repositorio; este script la regenera.
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses

from .extract_graph import RAW_DIR, SERVER

RING_TYPES = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG"]


def compute_eb_angles(dataset: str = "male-cns:v1.0", slug: str = "malecns") -> pd.DataFrame:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    client = Client(SERVER, dataset=dataset, token=os.environ["NEUPRINT_TOKEN"])
    out_dir = os.path.join(RAW_DIR, slug)
    nodes = pd.read_csv(os.path.join(out_dir, "nodes.csv"))
    ids = nodes.loc[nodes["type"].isin(RING_TYPES), "bodyId"].tolist()

    syn = fetch_synapses(NC(bodyId=ids), SC(rois=["EB"], primary_only=True), client=client)
    xyz = syn[["x", "y", "z"]].to_numpy(float)
    _, _, vt = np.linalg.svd(xyz - xyz.mean(0), full_matrices=False)
    plane = (xyz - xyz.mean(0)) @ vt[:2].T
    syn["ang"] = np.arctan2(plane[:, 1], plane[:, 0])

    rows = []
    for body_id, g in syn.groupby("bodyId"):
        z = (g["confidence"].to_numpy() * np.exp(1j * g["ang"].to_numpy())).sum() / g["confidence"].sum()
        rows.append((body_id, float(np.angle(z)), float(abs(z)), len(g)))
    angles = pd.DataFrame(rows, columns=["bodyId", "eb_angle", "concentration", "n_syn"])
    angles.to_csv(os.path.join(out_dir, "eb_angles.csv"), index=False)
    return angles


if __name__ == "__main__":
    a = compute_eb_angles()
    print(f"{len(a)} neuronas con ángulo EB; concentración mínima {a['concentration'].min():.2f}")

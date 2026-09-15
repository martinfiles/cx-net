"""
Carga el grafo de la Fase 1 (topología + metadatos de nodo) y lo convierte en
tensores de PyTorch, incluyendo una posición angular aproximada por neurona
para las poblaciones "brújula" (EPG/EPGt) y "conductoras" (PEN_a/PEN_b).

AVISO IMPORTANTE sobre `ring_angle`: la correspondencia real glomérulo-PB ->
cuña-EB no es una simple secuencia lineal (tiene un patrón de solapamiento
descrito en Turner-Evans et al. 2017 / Hulse et al. 2021 que no he
reproducido aquí de memoria para evitar afirmar un dato anatómico sin
verificar). La función `_ring_angle` usa una aproximación deliberada:
glomérulo 1-8 de cada hemisferio -> 8 posiciones equiespaciadas, hemisferios
L/R desplazados 180°, glomérulo 9 se pliega vía módulo. Es suficiente para
validar que el pipeline de entrenamiento funciona, pero debe contrastarse
contra la tabla real de Hulse et al. (2021, material suplementario) antes de
interpretar cualquier resultado cuantitativo de decodificación de rumbo.
"""

import math
import os
import re

import numpy as np
import pandas as pd
import torch

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
DATA_DIR = os.path.join(RAW_DIR, "malecns")  # dataset real; usar RAW_DIR/hemibrain para el piloto

COMPASS_TYPES = {"EPG", "EPGt"}
DRIVE_TYPES = {"PEN_a(PEN1)", "PEN_b(PEN2)"}
RING_POSITION_TYPES = COMPASS_TYPES | DRIVE_TYPES | {"PEG"}


def _parse_instance(instance: str) -> dict:
    if not isinstance(instance, str):
        return {"hemisphere": None, "glomerulus": None}
    glomeruli = re.findall(r"([LR])(\d+)", instance)
    trailing = re.search(r"_([LR])$", instance)
    if trailing:
        hemisphere = trailing.group(1)
    elif glomeruli:
        hemisphere = glomeruli[0][0]
    else:
        hemisphere = None
    glomerulus = int(glomeruli[0][1]) if glomeruli else None
    return {"hemisphere": hemisphere, "glomerulus": glomerulus}


def _ring_angle(row) -> float:
    """Ver aviso en el docstring del módulo: aproximación, no verificada."""
    if row["type"] not in RING_POSITION_TYPES or row["glomerulus"] is None:
        return float("nan")
    offset = 0 if row["hemisphere"] == "L" else 8
    idx = ((row["glomerulus"] - 1) % 8) + offset
    return 2 * math.pi * idx / 16


def load_cx_graph(data_dir: str = DATA_DIR):
    nodes = pd.read_csv(os.path.join(data_dir, "nodes.csv"))
    edges = pd.read_csv(os.path.join(data_dir, "graph_no_sign.csv"))

    parsed = nodes["instance"].apply(_parse_instance)
    nodes["hemisphere"] = parsed.apply(lambda d: d["hemisphere"])
    nodes["glomerulus"] = parsed.apply(lambda d: d["glomerulus"])
    nodes["ring_angle"] = nodes.apply(_ring_angle, axis=1)

    id_to_idx = {body_id: i for i, body_id in enumerate(nodes["bodyId"])}
    n_nodes = len(nodes)

    src = edges["source"].map(id_to_idx).to_numpy()
    dst = edges["target"].map(id_to_idx).to_numpy()
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    synapse_weight = torch.tensor(edges["weight"].to_numpy(), dtype=torch.float32)

    return {
        "nodes": nodes,
        "id_to_idx": id_to_idx,
        "edge_index": edge_index,
        "synapse_weight": synapse_weight,
        "n_nodes": n_nodes,
    }


if __name__ == "__main__":
    graph = load_cx_graph()
    nodes = graph["nodes"]
    print(f"Nodos: {graph['n_nodes']} | Aristas: {graph['edge_index'].shape[1]}")
    print(nodes.groupby("type")["hemisphere"].value_counts())
    print("Con ring_angle definido:", nodes["ring_angle"].notna().sum(), "/", len(nodes))

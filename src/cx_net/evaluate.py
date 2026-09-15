"""
Fase 4 — Evaluación de H1 y H2 sobre male-cns:v1.0.

H1: el signo aprendido por la red (sin haber visto nunca el neurotransmisor
real) coincide, más de lo esperable por azar, con el neurotransmisor real
anotado para la neurona de ORIGEN de cada sinapsis (el neurotransmisor es una
propiedad de la neurona, no de la sinapsis individual -- ver lab-notebook,
entrada 2026-09-16 (6)).

Control estadístico: test de permutación de etiquetas -- se baraja qué
neurona tiene qué neurotransmisor (preservando el recuento real: 110
colinérgicas, 42 glutamatérgicas) muchas veces, y se recalcula el acuerdo
cada vez. Esto es DISTINTO del modelo nulo de Dhiman (2026), que baraja la
TOPOLOGÍA para una pregunta sobre velocidad de aprendizaje -- aquí la
pregunta es sobre acuerdo de signo, así que el control apropiado es permutar
la asignación de neurotransmisor, no la topología.

H2: ¿el acuerdo varía por tipo celular? (desglose, no requiere permutación).
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from .graph_utils import DATA_DIR, load_cx_graph
from .model import CXRingNetwork

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim")

# Acetilcolina: excitadora (nicotínica) en el SNC de Drosophila.
# Glutamato: inhibidor en este circuito (vía GluClalfa) -- ver lab-notebook.
NT_SIGN = {"acetylcholine": 1.0, "glutamate": -1.0}


def load_trained_model(graph: dict) -> CXRingNetwork:
    model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"])
    state = torch.load(os.path.join(INTERIM_DIR, "model_pilot.pt"))
    model.load_state_dict(state)
    return model


def attach_ground_truth(nodes: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    gt = pd.read_csv(os.path.join(data_dir, "ground_truth_nt.csv"))
    unknown = set(gt["predicted_nt"].unique()) - set(NT_SIGN.keys())
    if unknown:
        raise ValueError(f"Neurotransmisor sin mapeo de signo definido: {unknown}")
    merged = nodes.merge(gt[["bodyId", "predicted_nt"]], on="bodyId", how="left")
    if merged["predicted_nt"].isna().any():
        raise ValueError("Hay neuronas sin neurotransmisor anotado tras el merge.")
    merged["expected_sign"] = merged["predicted_nt"].map(NT_SIGN)
    return merged


def run_evaluation(n_permutations: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    graph = load_cx_graph()
    nodes = attach_ground_truth(graph["nodes"], DATA_DIR)
    model = load_trained_model(graph)

    learned_sign = model.learned_signs().numpy()  # [E], +-1
    src = graph["edge_index"][0].numpy()
    node_sign_true = nodes["expected_sign"].to_numpy()
    expected_sign = node_sign_true[src]

    observed_agreement = float((learned_sign == expected_sign).mean())

    # Test de permutación: barajar qué neurona tiene qué neurotransmisor.
    null_agreements = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(node_sign_true)
        expected_perm = permuted[src]
        null_agreements[i] = (learned_sign == expected_perm).mean()

    p_value = (np.sum(null_agreements >= observed_agreement) + 1) / (n_permutations + 1)

    # H2: desglose por tipo celular de la neurona de origen.
    breakdown = {}
    for cell_type in nodes["type"].unique():
        mask = nodes.loc[nodes["type"] == cell_type, "bodyId"]
        idx_set = set(nodes.index[nodes["type"] == cell_type])
        edge_mask = np.isin(src, list(idx_set))
        if edge_mask.sum() == 0:
            continue
        breakdown[cell_type] = {
            "n_edges": int(edge_mask.sum()),
            "agreement": float((learned_sign[edge_mask] == expected_sign[edge_mask]).mean()),
        }

    results = {
        "n_edges": int(len(learned_sign)),
        "observed_agreement": observed_agreement,
        "null_mean": float(null_agreements.mean()),
        "null_std": float(null_agreements.std()),
        "p_value": float(p_value),
        "h1_supported": bool(p_value < 0.05 and observed_agreement > null_agreements.mean()),
        "breakdown_by_type_h2": breakdown,
    }

    os.makedirs(INTERIM_DIR, exist_ok=True)
    with open(os.path.join(INTERIM_DIR, "h1_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_evaluation()
    print(json.dumps(results, indent=2))

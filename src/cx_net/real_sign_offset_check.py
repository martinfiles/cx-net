"""
Repite real_sign_task_check.py con dos correcciones (ver lab-notebook,
2026-09-19 (4)):

1. Pérdida invariante al desfase constante: la tarea original pide el rumbo
   ABSOLUTO (empieza en 0), pero la red no recibe ninguna pista de dónde está
   el 0, así que incluso un integrador perfecto tendría el bump en una fase
   arbitraria y daría pérdida ~1. Aquí se resta, por ensayo, el desfase
   circular medio entre el ángulo decodificado y el rumbo -- se mide si la red
   INTEGRA, no si ancla el cero.
2. Mapeos de `ring_angle` alternativos. Hulse et al. (2021) (Fig. 16 y texto):
   cada hemisferio del PB muestrea el anillo completo a ~45° (8 glomérulos) y
   L y R están desfasados 22.5°. El mapeo actual del proyecto ('actual')
   coloca L en media circunferencia y R en la otra media.
     - 'actual':  idx=(g-1)%8 + (0 si L, 8 si R), angulo=2*pi*idx/16
     - 'interl+': idx=2*((g-1)%8) + (0 si L, 1 si R)   (R = L + 22.5°)
     - 'interl-': igual, pero con el sentido invertido (angulo = -angulo)
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch

from .evaluate import INTERIM_DIR, attach_ground_truth, load_trained_model
from .graph_utils import DATA_DIR, RING_POSITION_TYPES, load_cx_graph
from .model import CXRingNetwork
from .task import build_external_input, decode_heading, generate_held_out_set

HOLD_PROB = 0.3


def set_ring_angle(nodes, mapping):
    ang = np.full(len(nodes), np.nan)
    for i, r in enumerate(nodes.itertuples()):
        if r.type not in RING_POSITION_TYPES or r.glomerulus is None or np.isnan(r.glomerulus):
            continue
        g0 = (int(r.glomerulus) - 1) % 8
        is_r = r.hemisphere == "R"
        if mapping == "actual":
            idx = g0 + (8 if is_r else 0)
            ang[i] = 2 * math.pi * idx / 16
        else:
            idx = 2 * g0 + (1 if is_r else 0)
            a = 2 * math.pi * idx / 16
            ang[i] = a if mapping == "interl+" else -a
    out = nodes.copy()
    out["ring_angle"] = ang
    return out


def offset_invariant_loss(decoded, heading, warmup=20):
    tgt = torch.as_tensor(heading, dtype=torch.float32)
    diff = decoded[warmup:] - tgt[warmup:]
    off = torch.atan2(torch.sin(diff).mean(), torch.cos(diff).mean())
    return (1 - torch.cos(diff - off)).mean().item()


def eval_signs(graph, nodes, signs, trials, soft_model=None):
    if soft_model is None:
        model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"])
        with torch.no_grad():
            model.sign_param.copy_(torch.as_tensor(signs, dtype=torch.float32) * 10.0)
    else:
        model = soft_model
    with torch.no_grad():
        tot = 0.0
        for av, heading in trials:
            states = model(build_external_input(av, nodes))
            tot += offset_invariant_loss(decode_heading(states, nodes), heading)
    return tot / len(trials)


def run(n_perm=150, seed=0):
    rng = np.random.default_rng(seed)
    graph = load_cx_graph()
    gt = attach_ground_truth(graph["nodes"], DATA_DIR)
    node_sign = gt["expected_sign"].to_numpy()
    src = graph["edge_index"][0].numpy()
    trials = generate_held_out_set(hold_prob=HOLD_PROB)
    out = {"config": dict(hold_prob=HOLD_PROB, n_perm=n_perm), "mappings": {}}

    for mapping in ("actual", "interl+", "interl-"):
        nodes = set_ring_angle(graph["nodes"], mapping)
        res = {"real": eval_signs(graph, nodes, node_sign[src], trials)}
        if mapping == "actual":  # los modelos entrenados usaron este decodificador
            soft = []
            for s in range(8):
                m = load_trained_model(graph, run_label=f"signreg05_seed{s}")
                soft.append(eval_signs(graph, nodes, None, trials, soft_model=m))
            res["trained_soft"] = soft
        t0 = time.time()
        res["shuffled"] = [eval_signs(graph, nodes, rng.permutation(node_sign)[src], trials)
                           for _ in range(n_perm)]
        pl = np.array(res["shuffled"])
        res["summary"] = dict(real=res["real"], shuf_mean=float(pl.mean()), shuf_std=float(pl.std()),
                              shuf_p1=float(np.percentile(pl, 1)), shuf_p5=float(np.percentile(pl, 5)),
                              p_real_le=float((np.sum(pl <= res["real"]) + 1) / (n_perm + 1)))
        print(mapping, res["summary"], f"({time.time()-t0:.0f}s)", flush=True)
        if "trained_soft" in res:
            print("  trained soft:", np.round(res["trained_soft"], 3), flush=True)
        out["mappings"][mapping] = res
        with open(os.path.join(INTERIM_DIR, "real_sign_offset_check.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run(n_perm=int(sys.argv[1]) if len(sys.argv) > 1 else 150)

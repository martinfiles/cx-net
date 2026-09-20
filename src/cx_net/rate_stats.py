"""
Histogramas de las tasas de activación por tipo celular en redes entrenadas de la
enumeración de patrones (entrada (13) del cuaderno): muestra que la solución con
la asignación real explota tasas negativas de Delta7 (con `tanh`).
Salida: results/rates_by_type.json. Requiere los modelos en data/interim/.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .graph_utils import load_cx_graph
from .model import CXRingNetwork
from .task import build_external_input, generate_held_out_set

MODELS = {"real": "realctl_tanh_p", "sin_inhibicion": "types_mask0_p", "solo_EPG_inhibe": "types_mask2_p"}
BINS = np.linspace(-1, 1, 41)


def main(n_trials=5, t_min=40):
    g = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    nodes = g["nodes"]
    tids = torch.as_tensor(pd.factorize(nodes["type"])[0])
    names = list(pd.factorize(nodes["type"])[1])
    trials = generate_held_out_set(hold_prob=0.3, anchor=True, max_av=0.15)[:n_trials]
    out = {"bins": BINS.tolist(), "types": names, "n_trials": n_trials, "t_min": t_min, "models": {}}
    for key, label in MODELS.items():
        m = CXRingNetwork(g["n_nodes"], g["edge_index"], g["synapse_weight"], tau=10.0, recurrent_gain=2.0,
                          activation="tanh", type_ids=tids, learn_type_params=True)
        m.load_state_dict(torch.load(os.path.join("data", "interim", f"model_{label}.pt")))
        with torch.no_grad():
            S = torch.cat([m(build_external_input(av, nodes, gain=10.0, cue_phase=t0, cue_gain=10.0))[t_min:]
                           for av, h, t0 in trials])
        out["models"][key] = {"label": label, "types": {}}
        for t in names:
            r = S[:, (nodes["type"] == t).to_numpy()].numpy().ravel()
            hist, _ = np.histogram(r, bins=BINS)
            out["models"][key]["types"][t] = {"hist": hist.tolist(), "frac_positive": float((r > 0).mean()),
                                              "mean": float(r.mean())}
    json.dump(out, open(os.path.join("results", "rates_by_type.json"), "w"))
    print({k: round(v["types"]["Delta7"]["frac_positive"], 2) for k, v in out["models"].items()})


if __name__ == "__main__":
    main()

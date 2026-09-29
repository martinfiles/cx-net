"""
¿Las redes reales ya entrenadas en la tarea vieja (blanca, max_av=0.15) sirven en la tarea OU
nueva (max_av=0.04, T=500, tau_v=50)? Sin reentrenar. Es un control de la 3ª revisión (punto A):
si transfieren, la tarea nueva es resoluble por esta clase de modelos a esa velocidad.
Salida: results/transfer_check_v3.json
"""
import json
import os

import pandas as pd
import torch

from .evaluate import INTERIM_DIR
from .graph_utils import load_cx_graph
from .model import CXRingNetwork
from .task import generate_held_out_set, integration_diagnostics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def main():
    graph = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    nodes = graph["nodes"]
    type_ids = torch.as_tensor(pd.factorize(nodes["type"])[0])
    trials = generate_held_out_set(n_trials=30, T=500, hold_prob=0.3, anchor=True, max_av=0.04,
                                   velocity_process="ou", tau_v=50.0)
    out = {}
    for lab in ["realctl_tanh_p", "dose_real_seed1_p", "dose_real_seed2_p"]:
        m = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"], tau=10.0,
                          recurrent_gain=2.0, activation="tanh", type_ids=type_ids, learn_type_params=True)
        m.load_state_dict(torch.load(os.path.join(INTERIM_DIR, f"model_{lab}.pt")))
        d = integration_diagnostics(m, nodes, trials, in_gain=10.0, cue_gain=10.0)
        out[lab] = d
        print(lab, {k: round(v, 3) for k, v in d.items()})
    json.dump(out, open(os.path.join(RESULTS_DIR, "transfer_check_v3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

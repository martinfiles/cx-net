"""
Comprobación (sin entrenar) de si la TAREA selecciona los signos reales.

Se fija el signo de cada arista al del neurotransmisor real de su neurona de
origen (|tanh|=1, magnitudes de sinapsis como siempre) y se mide la pérdida
held-out en la tarea de integración de rumbo (la del set de validación de los
modelos `signreg05`: hold_prob=0.3). Se compara con:
  - los modelos entrenados `signreg05` (signo blando, tal cual se guardaron y
    con el signo duro sign(sign_param) a |tanh|=1),
  - N asignaciones de neurotransmisor barajadas entre neuronas (mismo recuento
    110/42; control de permutación a nivel de neurona, como en evaluate.py),
  - el nivel de azar de la tarea (pérdida de un decodificador constante/aleatorio
    se estima aparte en el cuaderno, ~0.85-1.0).

Lectura: si los signos reales no dan mejor pérdida que los barajados, la
tarea no discrimina la química real (lectura 2 de la discusión). Si la dan
mucho mejor y los modelos entrenados no los recuperan, el fallo estaría en la
recuperación por entrenamiento.
"""

import json
import os
import sys
import time

import numpy as np
import torch

from .evaluate import INTERIM_DIR, attach_ground_truth, load_trained_model
from .graph_utils import DATA_DIR, load_cx_graph
from .model import CXRingNetwork
from .task import evaluate_on_trials, generate_held_out_set

HOLD_PROB = 0.3


def loss_with_signs(graph, nodes, signs, trials):
    model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"])
    with torch.no_grad():
        model.sign_param.copy_(torch.as_tensor(signs, dtype=torch.float32) * 10.0)
    return evaluate_on_trials(model, nodes, trials)


def run(n_perm: int = 100, n_seeds: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    graph = load_cx_graph()
    nodes = graph["nodes"]
    gt = attach_ground_truth(nodes, DATA_DIR)
    node_sign = gt["expected_sign"].to_numpy()
    src = graph["edge_index"][0].numpy()
    trials = generate_held_out_set(hold_prob=HOLD_PROB)

    out = {"config": dict(hold_prob=HOLD_PROB, n_perm=n_perm, n_seeds=n_seeds)}

    out["real_signs_loss"] = loss_with_signs(graph, nodes, node_sign[src], trials)
    print("real:", out["real_signs_loss"], flush=True)

    soft, hard = [], []
    for s in range(n_seeds):
        m = load_trained_model(graph, run_label=f"signreg05_seed{s}")
        soft.append(evaluate_on_trials(m, nodes, trials))
        hard.append(loss_with_signs(graph, nodes, m.learned_signs().numpy(), trials))
    out["trained_soft_loss"] = soft
    out["trained_hard_loss"] = hard
    print("trained soft:", np.round(soft, 3), "\ntrained hard:", np.round(hard, 3), flush=True)

    perm_losses = []
    t0 = time.time()
    for i in range(n_perm):
        perm_losses.append(loss_with_signs(graph, nodes, rng.permutation(node_sign)[src], trials))
        if i % 10 == 0:
            print(f"perm {i} ({time.time()-t0:.0f}s): {perm_losses[-1]:.3f}", flush=True)
    out["shuffled_losses"] = perm_losses
    pl = np.array(perm_losses)
    out["shuffled_summary"] = dict(mean=float(pl.mean()), std=float(pl.std()), min=float(pl.min()),
                                   max=float(pl.max()),
                                   p_real_le=float((np.sum(pl <= out["real_signs_loss"]) + 1) / (n_perm + 1)))
    print(out["shuffled_summary"])

    with open(os.path.join(INTERIM_DIR, "real_sign_task_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run(n_perm=int(sys.argv[1]) if len(sys.argv) > 1 else 100)

"""
Búsqueda del régimen dinámico (sin entrenar): ¿existe algún valor de
`recurrent_gain`, `tau`, ganancia de entrada y ganancia de la pista con el que
la red con los SIGNOS REALES fijos integre el rumbo en la tarea anclada?

Criterio A (decidido con el usuario, 2026-09-19; ver lab-notebook (7)): el
régimen se elige de modo que la tarea sea realizable con la química real. El
aprendiz nunca ve el neurotransmisor -- solo verá los hiperparámetros elegidos --
pero como el régimen se selecciona con ayuda del ground truth, es un grado de
libertad que debe declararse en el preprint.

Para no sobreajustar el held-out (semillas 900M) la búsqueda usa un conjunto de
AJUSTE distinto (semillas TUNING_SEED_BASE), reservando el held-out para la
evaluación final.

Métricas por configuración (signos reales, ambos sentidos de giro): pérdida
absoluta en el conjunto de ajuste, y pendiente = sum(dd*hc)/sum(hc*hc) entre el
cambio decodificado dd y el cambio real hc desde el paso 20 (1 = integra a
velocidad correcta, 0 = no integra).
"""

import itertools
import os

import numpy as np
import pandas as pd
import torch

from .evaluate import INTERIM_DIR, attach_ground_truth
from .graph_utils import DATA_DIR, load_cx_graph
from .model import CXRingNetwork
from .task import build_external_input, circular_loss, decode_heading, generate_anchored_trial

TUNING_SEED_BASE = 800_000_000
MAX_AV = 0.15
HOLD_PROB = 0.3


def tuning_set(n=20):
    return [generate_anchored_trial(max_av=MAX_AV, hold_prob=HOLD_PROB, seed=TUNING_SEED_BASE + i) for i in range(n)]


def evaluate_regime(graph, signs, trials, rec_gain, tau, in_gain, cue_gain, activation="tanh"):
    model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"],
                          tau=tau, recurrent_gain=rec_gain, activation=activation)
    with torch.no_grad():
        model.sign_param.copy_(torch.as_tensor(signs, dtype=torch.float32) * 10.0)
        nodes = graph["nodes"]
        losses, num, den, r_max = [], 0.0, 0.0, 0.0
        for av, heading, theta0 in trials:
            ext = build_external_input(av, nodes, gain=in_gain, cue_phase=theta0, cue_gain=cue_gain)
            st = model(ext)
            dec = decode_heading(st, nodes)
            losses.append(circular_loss(dec, heading).item())
            d = np.unwrap(dec.numpy())
            dd = d[20:] - d[20]
            hc = heading[20:] - heading[20]
            num += float((dd * hc).sum())
            den += float((hc * hc).sum())
    return float(np.mean(losses)), num / max(den, 1e-9)


# Ronda 1: rejilla amplia. Ronda 2 (tras ver la ronda 1): zona de tau grande y
# ganancias de entrada altas, porque con tau mayor el bump se mueve más lento.
GRIDS = {
    "regime_search.csv": ([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], [2.0, 5.0, 10.0], [1.0, 3.0, 10.0, 30.0], [3.0, 10.0]),
    "regime_search_r2.csv": ([0.75, 1.0, 1.5, 2.0, 3.0], [10.0, 20.0], [3.0, 10.0, 30.0, 100.0, 300.0], [10.0]),
    # Ronda 3: activación rectificada (tasas >= 0), rejilla propia.
    "regime_search_r3_rect.csv": ([1.0, 2.0, 4.0, 8.0, 16.0], [2.0, 5.0, 10.0, 20.0], [1.0, 3.0, 10.0, 30.0, 100.0], [10.0]),
}


def run(n_trials=20, out_name="regime_search.csv"):
    grid = list(itertools.product(*GRIDS[out_name]))
    activation = "rectified" if "rect" in out_name else "tanh"
    trials = tuning_set(n_trials)
    rows = []
    for ring_sign in (+1.0, -1.0):
        graph = load_cx_graph(ring_source="eb_synapses", ring_sign=ring_sign)
        nodes = attach_ground_truth(graph["nodes"], DATA_DIR)
        signs = nodes["expected_sign"].to_numpy()[graph["edge_index"][0].numpy()]
        for k, (rg, tau, ig, cg) in enumerate(grid):
            loss, slope = evaluate_regime(graph, signs, trials, rg, tau, ig, cg, activation)
            rows.append(dict(ring_sign=ring_sign, rec_gain=rg, tau=tau, in_gain=ig, cue_gain=cg,
                             loss=loss, slope=slope))
            if k % 24 == 0:
                print(f"ring_sign={ring_sign:+.0f} {k}/{len(grid)} ultimo: rg={rg} tau={tau} ig={ig} cg={cg} "
                      f"loss={loss:.3f} slope={slope:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(INTERIM_DIR, out_name), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = run(out_name=sys.argv[1] if len(sys.argv) > 1 else "regime_search.csv")
    print(df.sort_values("loss").head(15).to_string())

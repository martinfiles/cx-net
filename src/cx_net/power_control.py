"""
Control positivo de potencia para el test de H1 (test de permutación de
etiqueta de neurotransmisor, ver evaluate.py).

Pregunta: si el signo aprendido SÍ contuviera información sobre el
neurotransmisor real, ¿con qué frecuencia lo detectaría este test, dadas las
9.160 aristas reales, las etiquetas reales y los signos reales de los modelos
`sign_reg`?

Método: a los signos aprendidos de cada modelo se les "siembra" señal
conocida. Una fracción q de aristas (modo `edge`) o de neuronas de origen
(modo `neuron`, todas sus aristas salientes) pasa a tener el signo del
neurotransmisor real; el resto conserva el signo aprendido. Se repite el
mismo test que en evaluate.py (2000 permutaciones, una cola, alfa=0.05) y se
mide la tasa de detección.

Dos variantes de "verdad" contra la que se siembra y se testea:
- `real`: las etiquetas reales. Hereda cualquier señal residual real de los
  modelos (p. ej. la semilla 1) y, con q=0, las repeticiones son idénticas
  (permutaciones fijas), así que q=0 NO es una tasa de falsos positivos.
- `decoy`: una asignación de etiquetas barajada, sorteada de nuevo en cada
  repetición. No puede contener señal real, así que q=0 mide el falso
  positivo (debe ser ~alfa) y q>0 mide potencia limpia.

El modo `neuron` es el realista: el neurotransmisor es propiedad de la
neurona, así que una red que lo aprendiera lo haría por neuronas, y la
permutación (que baraja a nivel de neurona) tiene menos unidades
independientes que aristas.
"""

import json
import os

import numpy as np

from .evaluate import INTERIM_DIR, attach_ground_truth, load_trained_model
from .graph_utils import DATA_DIR, load_cx_graph

Q_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]


def run_power_control(
    run_prefix: str = "signreg05",
    n_seeds: int = 8,
    n_permutations: int = 2000,
    n_reps: int = 200,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    graph = load_cx_graph()
    nodes = attach_ground_truth(graph["nodes"], DATA_DIR)
    src = graph["edge_index"][0].numpy()
    node_sign = nodes["expected_sign"].to_numpy()
    n_edges, n_nodes = len(src), len(node_sign)
    expected = node_sign[src]

    # Permutaciones de etiqueta fijas (compartidas por todos los modelos).
    perm_expected = np.stack(
        [rng.permutation(node_sign)[src] for _ in range(n_permutations)]
    ).astype(np.float32)

    def agreement_and_p(signs, target):
        s = signs.astype(np.float32)
        obs = float((s == target).mean())
        null = (perm_expected @ s) / n_edges * 0.5 + 0.5  # acuerdo = (1 + <s,e>/E)/2
        p = (np.sum(null >= obs - 1e-12) + 1) / (n_permutations + 1)
        return obs, float(null.mean()), p

    out = {"config": dict(run_prefix=run_prefix, n_seeds=n_seeds,
                          n_permutations=n_permutations, n_reps=n_reps, q_grid=Q_GRID),
           "results": {}}

    for truth, mode in [(t, m) for t in ("decoy", "real") for m in ("edge", "neuron")]:
        for q in Q_GRID:
            if truth == "real" and q == 0.0:
                continue
            agree, detected = [], []
            for s in range(n_seeds):
                model = load_trained_model(graph, run_label=f"{run_prefix}_seed{s}")
                learned = model.learned_signs().numpy()
                for _ in range(n_reps):
                    t_node = rng.permutation(node_sign) if truth == "decoy" else node_sign
                    t_edge = t_node[src]
                    if mode == "edge":
                        mask = rng.random(n_edges) < q
                    else:
                        mask = (rng.random(n_nodes) < q)[src]
                    seeded = np.where(mask, t_edge, learned)
                    obs, _, p = agreement_and_p(seeded, t_edge)
                    agree.append(obs)
                    detected.append(p < 0.05)
            key = f"{truth}_{mode}_q{q:.2f}"
            out["results"][key] = {
                "truth": truth, "mode": mode, "q": q,
                "mean_agreement": float(np.mean(agree)),
                "power": float(np.mean(detected)),
                "n_tests": len(detected),
            }
            print(f"{key}: agreement={np.mean(agree):.4f} power={np.mean(detected):.3f}", flush=True)

    with open(os.path.join(INTERIM_DIR, "power_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run_power_control()

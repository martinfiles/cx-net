"""
Magnitudes libres por arista con signos fijos (respuesta a la revisión externa, entrada
2026-09-21 del cuaderno). Compara, con la misma receta (600 épocas, tanh, ganancias por tipo):
  - baseline por tipo (sin ganancia por arista): `realctl_tanh_p`, `dose_real_seed{1,2}_p`,
    `realctl_shuf{1..6}_tanh_p` (mismos barajados por neurona que los `edgegain_shuf{k}_p`),
  - con ganancia positiva por arista (`edgegain_*`): real (3 semillas), 6 barajados, sin
    inhibición (mask0) y solo EPG (mask2).
Además mide la poda: una ganancia por arista < 0.1 cuenta como arista podada; se separan las
aristas con el signo del neurotransmisor anotado de las de signo distinto (solo hay de estas en
los barajados). Si los barajados «convergen» podando las aristas de signo distinto, la libertad
de magnitud vuelve no identificable el signo (y el resultado no dice nada de H1).
Salida: results/edge_gain_analysis.json. Requiere data/interim (no versionado).
"""
import json
import os

import numpy as np
import torch

from .evaluate import attach_ground_truth
from .graph_utils import DATA_DIR, load_cx_graph

INTERIM = os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def load(label):
    p = os.path.join(INTERIM, f"results_{label}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"]
    d = r.get("diagnostics") or {}
    return {"label": label, "held_out": r["held_out_loss"], "slope": d.get("slope"), "anchor_err_deg": d.get("anchor_err_deg"),
            "integrates": bool(r["held_out_loss"] < 0.466 and (d.get("slope") or 0) > 0.5)}


def main(prefix="edgegain", out_name="edge_gain_analysis.json"):
    g = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    nodes = g["nodes"]
    true_edge_sign = attach_ground_truth(nodes, DATA_DIR)["expected_sign"].to_numpy()[g["edge_index"][0].numpy()]
    out = {"baseline_per_type": {}, "edge_gain": {}}
    for label in ["realctl_tanh_p", "dose_real_seed1_p", "dose_real_seed2_p"] + [f"realctl_shuf{k}_tanh_p" for k in range(1, 7)]:
        r = load(label)
        if r:
            out["baseline_per_type"][label] = r
    for label in [f"{prefix}_real_seed{s}_p" for s in range(3)] + [f"{prefix}_shuf{k}_p" for k in range(1, 7)] + \
                 [f"{prefix}_mask0_p", f"{prefix}_mask2_p"]:
        r = load(label)
        if not r:
            continue
        sd = torch.load(os.path.join(INTERIM, f"model_{label}.pt"), weights_only=True)
        sign = np.sign(sd["sign_param"].numpy())
        gain = np.exp(np.clip(sd["log_edge_gain"].numpy(), -5, 5))
        wrong = sign != true_edge_sign
        pruned = gain < 0.1
        mag = g["synapse_weight"].abs()
        dst = g["edge_index"][1]
        tot = torch.zeros(g["n_nodes"]).index_add(0, dst, mag).clamp(min=1.0)
        w = (mag / tot[dst]).numpy()  # misma normalización por neurona destino que el modelo
        eff = w * gain
        r.update({
            "frac_edges_wrong_sign": float(wrong.mean()),
            "frac_pruned_all": float(pruned.mean()),
            "frac_pruned_wrong_sign": float(pruned[wrong].mean()) if wrong.any() else None,
            "frac_pruned_right_sign": float(pruned[~wrong].mean()),
            "mass_wrong_sign_before": float(w[wrong].sum() / w.sum()),
            "mass_wrong_sign_after": float(eff[wrong].sum() / eff.sum()),
            "gain_median": float(np.median(gain)), "gain_p05_p95": [float(np.percentile(gain, 5)), float(np.percentile(gain, 95))],
        })
        out["edge_gain"][label] = r
        print(f"{label:26s} held-out={r['held_out']:.3f} pendiente={r['slope']} integra={r['integrates']} "
              f"aristas signo distinto={r['frac_edges_wrong_sign']:.2f} podadas(signo distinto)={r['frac_pruned_wrong_sign']} "
              f"masa signo distinto {r['mass_wrong_sign_before']:.2f}->{r['mass_wrong_sign_after']:.2f}")
    json.dump(out, open(os.path.join(OUT, out_name), "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])

"""Análisis preespecificado del refuerzo de suficiencia funcional (entrada (10) del cuaderno)."""
import glob
import json
import os

import numpy as np

from .evaluate import INTERIM_DIR

MEMORY = 0.466  # referencia "recordar theta0 sin integrar" en el held-out


def load(label):
    r = json.load(open(os.path.join(INTERIM_DIR, f"results_{label}.json")))["results"]
    d = r["diagnostics"]
    return dict(label=label, held_out=r["held_out_loss"], slope=d["slope"], anchor=d["anchor_err_deg"],
                frac_nodes=r.get("frac_nodes_changed"), frac_edges=r.get("frac_edges_changed"),
                nominal=r.get("control_swap_fraction"))


def _fill_shuffle_changes(shuf):
    """Los barajados 1-4 se lanzaron antes de registrar frac_*_changed: se recalculan
    de la semilla de barajado (mismo procedimiento que train.py)."""
    from .evaluate import attach_ground_truth
    from .graph_utils import DATA_DIR, load_cx_graph
    g = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    truth = attach_ground_truth(g["nodes"], DATA_DIR)["expected_sign"].to_numpy()
    src = g["edge_index"][0].numpy()
    for x in shuf:
        k = int(x["label"].split("shuf")[1].split("_")[0])
        perm = np.random.default_rng(k).permutation(truth)
        x["frac_nodes"] = float((perm != truth).mean())
        x["frac_edges"] = float((perm[src] != truth[src]).mean())


def integra(x):
    return x["held_out"] < MEMORY and x["slope"] > 0.5


def main():
    real = [load("realctl_tanh_p")] + [load(f"dose_real_seed{s}_p") for s in (1, 2)]
    shuf = [load(f"realctl_shuf{k}_tanh_p") for k in range(1, 13)]
    _fill_shuffle_changes(shuf)
    swaps = [load(os.path.basename(p)[len("results_"):-len(".json")])
             for p in sorted(glob.glob(os.path.join(INTERIM_DIR, "results_dose_swap*_p.json")))]
    print("REAL (n=3):", [(round(x["held_out"], 3), round(x["slope"], 2)) for x in real])
    print(f"  integran: {sum(map(integra, real))}/3 | media held-out {np.mean([x['held_out'] for x in real]):.3f}")
    ho = np.array([x["held_out"] for x in shuf])
    print(f"BARAJADOS (n=12): held-out min {ho.min():.3f} media {ho.mean():.3f} max {ho.max():.3f} | "
          f"pendiente max {max(x['slope'] for x in shuf):.2f} | integran: {sum(map(integra, shuf))}/12")
    m = np.mean([x["held_out"] for x in real])
    p_mean = (1 + np.sum(ho <= m)) / (len(ho) + 1)
    p_max = (1 + np.sum(ho <= max(x["held_out"] for x in real))) / (len(ho) + 1)
    print(f"p (media de reales) = {p_mean:.3f} | p (peor real) = {p_max:.3f}")
    print("\nINTERCAMBIO PARCIAL (nominal -> aristas cambiadas reales):")
    for f in sorted({x["nominal"] for x in swaps}):
        g = [x for x in swaps if x["nominal"] == f]
        print(f"  nominal {f:.2f}: aristas cambiadas {[round(x['frac_edges'], 3) for x in g]} | neuronas cambiadas "
              f"{[round(x['frac_nodes'], 3) for x in g]}\n     held-out {[round(x['held_out'], 3) for x in g]} pendiente "
              f"{[round(x['slope'], 2) for x in g]} | integran {sum(map(integra, g))}/{len(g)}")
    print("\nBARAJADOS: aristas cambiadas:", [round(x["frac_edges"], 3) for x in shuf])
    json.dump(dict(real=real, shuffled=shuf, swaps=swaps), open(os.path.join(INTERIM_DIR, "dose_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""
Reúne los resultados que respaldan el borrador en `results/` (versionado), porque
`data/interim/` está fuera del repositorio. Genera:
  - results/runs_summary.csv : una fila por corrida de entrenamiento (held-out,
    diagnósticos de integración, configuración clave).
  - copia de los JSON/CSV de análisis pequeños (resúmenes, no series por época).
  - results/epg_glomerulus_angles.csv : ángulo EB medio por (hemisferio, glomérulo)
    de los EPG (agregado; no contiene datos por neurona).
"""
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

from .evaluate import INTERIM_DIR

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "results")

COPY = ["types_analysis.json", "dose_analysis.json", "regime_search.csv", "regime_search_r2.csv",
        "regime_search_r3_rect.csv", "power_control.json", "real_sign_task_check.json",
        "real_sign_offset_check.json", "signreg05_sweep_summary_8seeds.json"] + [
        f"h1_evaluation_signreg05_seed{s}.json" for s in range(8)]

KEYS = ["n_epochs", "seed", "hold_prob", "sign_reg", "anchor", "max_av", "ring_source", "ring_sign",
        "activation", "type_params", "real_sign_control", "control_shuffle_seed", "control_swap_fraction",
        "control_type_mask", "frac_edges_changed", "recurrent_gain", "tau", "in_gain", "cue_gain",
        "mean_abs_sign", "frac_polarized_gt_0.9"]


def runs_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(INTERIM_DIR, "results_*.json"))):
        r = json.load(open(p)).get("results")
        if not r or "held_out_loss" not in r:
            continue
        d = r.get("diagnostics") or {}
        rows.append({"label": r.get("run_label", os.path.basename(p)[8:-5]), "held_out_loss": r["held_out_loss"],
                     "slope": d.get("slope"), "anchor_err_deg": d.get("anchor_err_deg"),
                     **{k: r.get(k) for k in KEYS}})
    return pd.DataFrame(rows)


def glomerulus_angles():
    from .graph_utils import DATA_DIR, load_cx_graph
    g = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    n = g["nodes"]
    e = n[n["type"] == "EPG"]
    rows = []
    for (h, gl), grp in e.groupby(["hemisphere", "glomerulus"]):
        z = np.exp(1j * grp["ring_angle"].to_numpy()).mean()
        rows.append({"hemisphere": h, "glomerulus": int(gl), "n_neurons": len(grp),
                     "mean_angle_deg": float(np.degrees(np.angle(z)) % 360)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in COPY:
        shutil.copy(os.path.join(INTERIM_DIR, f), os.path.join(OUT, f))
    t = runs_table()
    t.to_csv(os.path.join(OUT, "runs_summary.csv"), index=False)
    glomerulus_angles().to_csv(os.path.join(OUT, "epg_glomerulus_angles.csv"), index=False)
    print(f"{len(t)} corridas en runs_summary.csv; {len(COPY)} ficheros copiados")


if __name__ == "__main__":
    main()

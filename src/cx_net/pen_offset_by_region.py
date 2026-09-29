"""
Desfase de fase de las PEN, recalculado separando las regiones (respuesta a la
revisión externa, sección 5 / "trabajo futuro" de `docs/preprint-discussion.md`).

Contexto: `extract_graph.build_topology_only_graph` suma los pesos de sinapsis
de TODAS las regiones (PB, EB, resto) al construir `graph_no_sign.csv`; la
validación anterior del mapeo de fase con los PEN (lab-notebook, entrada
2026-09-19 (5): "≈ -6° / +5.5°") usaba esa misma mezcla. `fetch_adjacencies`
ya devuelve una columna `roi` por sinapsis que ese paso descartaba.

Método: para cada neurona PEN, se calculan dos fases usando SOLO sinapsis de
una región cada una (nunca mezcladas):
  - fase de entrada: media circular (ponderada por nº de sinapsis) del
    `eb_angle` medido (`extract_eb_angles.py`) de sus EPG/EPGt presinápticas,
    restringido a sinapsis en el PB.
  - fase de salida: lo mismo con sus EPG/EPGt postsinápticas, restringido al EB.
El desfase de esa PEN es la diferencia circular (salida - entrada). Se
compara también con la fase propia de la PEN (medida directamente de sus
propias sinapsis en el EB, `eb_angles.csv`), que no depende de conectividad.

Salida: results/pen_offset_by_region.json
"""
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies

from .extract_graph import CX_ROIS, SERVER

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "malecns")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

COMPASS_TYPES = {"EPG", "EPGt"}
PEN_TYPES = {"PEN_a(PEN1)": "PEN_a", "PEN_b(PEN2)": "PEN_b"}


def circular_weighted_mean(angles, weights):
    if len(angles) == 0 or np.sum(weights) == 0:
        return float("nan")
    z = np.sum(weights * np.exp(1j * np.asarray(angles)))
    return float(np.angle(z))


def circular_diff_deg(a, b):
    """(a - b) en grados, en (-180, 180]."""
    d = np.degrees(np.angle(np.exp(1j * (a - b))))
    return float(d)


def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    client = Client(SERVER, dataset="male-cns:v1.0", token=os.environ["NEUPRINT_TOKEN"])

    nodes = pd.read_csv(os.path.join(RAW_DIR, "nodes.csv"))
    eb_angles = pd.read_csv(os.path.join(RAW_DIR, "eb_angles.csv")).set_index("bodyId")["eb_angle"]
    type_of = dict(zip(nodes["bodyId"], nodes["type"]))

    ids = nodes["bodyId"].tolist()
    print("Consultando adyacencias con desglose por roi...")
    _, conn = fetch_adjacencies(sources=ids, targets=ids, client=client)
    conn["pre_type"] = conn["bodyId_pre"].map(type_of)
    conn["post_type"] = conn["bodyId_post"].map(type_of)

    results = {}
    for pen_col, pen_label in PEN_TYPES.items():
        pen_ids = nodes.loc[nodes["type"] == pen_col, "bodyId"]
        rows = []
        for pid in pen_ids:
            if pid not in eb_angles.index:
                continue
            in_edges = conn[(conn["bodyId_post"] == pid) & conn["pre_type"].isin(COMPASS_TYPES)
                             & (conn["roi"] == "PB")]
            out_edges = conn[(conn["bodyId_pre"] == pid) & conn["post_type"].isin(COMPASS_TYPES)
                              & (conn["roi"] == "EB")]
            in_ang = [eb_angles[b] for b in in_edges["bodyId_pre"] if b in eb_angles.index]
            in_w = [w for b, w in zip(in_edges["bodyId_pre"], in_edges["weight"]) if b in eb_angles.index]
            out_ang = [eb_angles[b] for b in out_edges["bodyId_post"] if b in eb_angles.index]
            out_w = [w for b, w in zip(out_edges["bodyId_post"], out_edges["weight"]) if b in eb_angles.index]
            if not in_ang or not out_ang:
                continue
            phase_in = circular_weighted_mean(in_ang, np.array(in_w, dtype=float))
            phase_out = circular_weighted_mean(out_ang, np.array(out_w, dtype=float))
            own = float(eb_angles[pid])
            rows.append({
                "bodyId": int(pid), "hemisphere": nodes.loc[nodes.bodyId == pid, "instance"].iloc[0][-2],
                "n_in_pb": int(sum(in_w)), "n_out_eb": int(sum(out_w)),
                "offset_out_minus_in_deg": circular_diff_deg(phase_out, phase_in),
                "own_minus_in_deg": circular_diff_deg(own, phase_in),
            })
        df = pd.DataFrame(rows)
        by_hemi = {}
        for h, g in df.groupby("hemisphere"):
            by_hemi[h] = {
                "n_neurons": len(g),
                "mean_offset_out_minus_in_deg": float(g["offset_out_minus_in_deg"].mean()),
                "mean_own_minus_in_deg": float(g["own_minus_in_deg"].mean()),
            }
        results[pen_label] = {"n_with_both_sides": len(df), "by_hemisphere": by_hemi,
                               "per_neuron": rows}
        print(f"{pen_label}: {len(df)} neuronas con entrada PB y salida EB medibles")
        for h, s in by_hemi.items():
            print(f"  hemisferio {h}: n={s['n_neurons']} "
                  f"desfase salida-entrada medio={s['mean_offset_out_minus_in_deg']:+.1f}° "
                  f"(propia-entrada={s['mean_own_minus_in_deg']:+.1f}°)")

    json.dump(results, open(os.path.join(RESULTS_DIR, "pen_offset_by_region.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

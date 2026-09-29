"""
Respuesta a la 3ª revisión externa (2026-09-29), punto B: diagnóstico del grafo
mezclado sin reentrenar.

1) Desfase de los PEN SIN filtrar por región (mismo cálculo que
   `pen_offset_by_region.py` pero usando las sinapsis de todas las regiones para
   la entrada y la salida). Si da ~50°, la explicación "el grafo mezclado diluye
   el desfase" es falsa.
2) Peso sináptico por región y por par de tipos (origen->destino).
3) Ablación por región sobre los modelos YA entrenados con la asignación real
   (3 semillas, grafo completo): se pone a cero la contribución de las
   sinapsis de una región (manteniendo la normalización original por neurona
   destino) y se re-evalúa held-out y diagnósticos con el mismo set de validación.

La consulta a neuPrint se cachea en data/interim/adjacency_by_roi.csv.
Salida: results/region_analysis.json
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from .evaluate import INTERIM_DIR
from .graph_utils import DATA_DIR, load_cx_graph
from .model import CXRingNetwork
from .pen_offset_by_region import COMPASS_TYPES, PEN_TYPES, circular_diff_deg, circular_weighted_mean
from .task import evaluate_on_trials, generate_held_out_set, integration_diagnostics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
CACHE = os.path.join(INTERIM_DIR, "adjacency_by_roi.csv")
REAL_LABELS = ["realctl_tanh_p", "dose_real_seed1_p", "dose_real_seed2_p"]
MEMORY = 0.466


def get_adjacency():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    from neuprint import Client, fetch_adjacencies
    from .extract_graph import SERVER
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    client = Client(SERVER, dataset="male-cns:v1.0", token=os.environ["NEUPRINT_TOKEN"])
    nodes = pd.read_csv(os.path.join(DATA_DIR, "nodes.csv"))
    ids = nodes["bodyId"].tolist()
    _, conn = fetch_adjacencies(sources=ids, targets=ids, client=client)
    conn.to_csv(CACHE, index=False)
    return conn


def pen_offsets(conn, nodes, eb_angles, type_of, in_rois, out_rois):
    out = {}
    for pen_col, label in PEN_TYPES.items():
        rows = []
        for pid in nodes.loc[nodes["type"] == pen_col, "bodyId"]:
            if pid not in eb_angles.index:
                continue
            ie = conn[(conn.bodyId_post == pid) & conn.bodyId_pre.map(type_of).isin(COMPASS_TYPES)]
            oe = conn[(conn.bodyId_pre == pid) & conn.bodyId_post.map(type_of).isin(COMPASS_TYPES)]
            if in_rois is not None:
                ie = ie[ie.roi.isin(in_rois)]
            if out_rois is not None:
                oe = oe[oe.roi.isin(out_rois)]
            ie = ie[ie.bodyId_pre.isin(eb_angles.index)].groupby("bodyId_pre")["weight"].sum()
            oe = oe[oe.bodyId_post.isin(eb_angles.index)].groupby("bodyId_post")["weight"].sum()
            if ie.empty or oe.empty:
                continue
            pin = circular_weighted_mean(eb_angles[ie.index].to_numpy(), ie.to_numpy(float))
            pout = circular_weighted_mean(eb_angles[oe.index].to_numpy(), oe.to_numpy(float))
            rows.append(circular_diff_deg(pout, pin))
        rows = np.array(rows)
        out[label] = {"n": len(rows), "mean_deg": float(np.mean(rows)),
                      "mean_abs_deg": float(np.mean(np.abs(rows))),
                      "frac_abs_between_40_60": float(np.mean((np.abs(rows) > 40) & (np.abs(rows) < 60)))}
    return out


def ablate(graph, conn, region_set, label):
    """Modelo real entrenado con la contribución de `region_set` puesta a cero."""
    nodes = graph["nodes"]
    idx = graph["id_to_idx"]
    src, dst = graph["edge_index"].numpy()
    total = pd.Series(graph["synapse_weight"].numpy(), index=pd.MultiIndex.from_arrays([src, dst]))
    c = conn.assign(s=conn.bodyId_pre.map(idx), d=conn.bodyId_post.map(idx))
    lost = c[c.roi.isin(region_set)].groupby(["s", "d"])["weight"].sum()
    frac_kept = (1 - lost.reindex(total.index).fillna(0) / total).clip(lower=0).to_numpy()
    type_ids = torch.as_tensor(pd.factorize(nodes["type"])[0])
    model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"], tau=10.0,
                          recurrent_gain=2.0, activation="tanh", type_ids=type_ids, learn_type_params=True)
    model.load_state_dict(torch.load(os.path.join(INTERIM_DIR, f"model_{label}.pt")))
    model.synapse_weight *= torch.as_tensor(frac_kept, dtype=torch.float32)
    return model


def main():
    conn = get_adjacency()
    nodes_raw = pd.read_csv(os.path.join(DATA_DIR, "nodes.csv"))
    eb_angles = pd.read_csv(os.path.join(DATA_DIR, "eb_angles.csv")).set_index("bodyId")["eb_angle"]
    type_of = dict(zip(nodes_raw["bodyId"], nodes_raw["type"]))
    res = {"rois": sorted(conn.roi.unique().tolist())}
    print("ROIs:", res["rois"])

    res["pen_offset"] = {
        "sin_filtrar": pen_offsets(conn, nodes_raw, eb_angles, type_of, None, None),
        "PB_entrada_EB_salida": pen_offsets(conn, nodes_raw, eb_angles, type_of, {"PB"}, {"EB"}),
        "solo_PB": pen_offsets(conn, nodes_raw, eb_angles, type_of, {"PB"}, {"PB"}),
        "solo_EB": pen_offsets(conn, nodes_raw, eb_angles, type_of, {"EB"}, {"EB"}),
    }
    print(json.dumps(res["pen_offset"], indent=1))

    conn = conn.assign(pre_type=conn.bodyId_pre.map(type_of), post_type=conn.bodyId_post.map(type_of))
    tot = conn["weight"].sum()
    res["weight_share_by_roi"] = (conn.groupby("roi")["weight"].sum() / tot).round(4).to_dict()
    print("peso por roi:", res["weight_share_by_roi"])
    pair = conn.pivot_table(index=["pre_type", "post_type"], columns="roi", values="weight", aggfunc="sum",
                            fill_value=0)
    pair["total"] = pair.sum(axis=1)
    pair = pair.sort_values("total", ascending=False)
    res["weight_by_type_pair_and_roi"] = {f"{a}->{b}": {k: int(v) for k, v in row.items() if v}
                                          for (a, b), row in pair.iterrows()}
    conn = conn.drop(columns=["pre_type", "post_type"])

    graph = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    trials = generate_held_out_set(T=200, hold_prob=0.3, perturb_amp=0.0, anchor=True, max_av=0.15,
                                   velocity_process="white", tau_v=15.0)
    conditions = {"ninguna": set()}
    for r in res["rois"]:
        if r not in ("PB", "EB"):
            conditions[r] = {r}
    conditions["todas_menos_PB_EB"] = set(res["rois"]) - {"PB", "EB"}
    conditions["PB"] = {"PB"}
    conditions["EB"] = {"EB"}
    res["ablation"] = {}
    for name, rs in conditions.items():
        rows = []
        for lab in REAL_LABELS:
            m = ablate(graph, conn, rs, lab)
            ho = evaluate_on_trials(m, graph["nodes"], trials, in_gain=10.0, cue_gain=10.0)
            d = integration_diagnostics(m, graph["nodes"], trials, in_gain=10.0, cue_gain=10.0)
            rows.append({"run": lab, "held_out": ho, "slope": d["slope"], "anchor_err_deg": d["anchor_err_deg"],
                         "integra": bool(ho < MEMORY and d["slope"] > 0.5)})
        res["ablation"][name] = {"runs": rows, "n_integra": sum(r["integra"] for r in rows),
                                 "mean_held_out": float(np.mean([r["held_out"] for r in rows])),
                                 "mean_slope": float(np.mean([r["slope"] for r in rows]))}
        a = res["ablation"][name]
        print(f"ablación {name:28s} held-out {a['mean_held_out']:.3f} pendiente {a['mean_slope']:.2f} "
              f"integra {a['n_integra']}/3", flush=True)
    json.dump(res, open(os.path.join(RESULTS_DIR, "region_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

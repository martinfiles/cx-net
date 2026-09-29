"""
Construye, desde la caché de adyacencias por ROI (`region_analysis.py`, sin red), dos grafos de
la 3ª revisión (punto B): `malecns_no_NO` (grafo completo SIN las sinapsis del NO) y
`malecns_pb_eb_no` (PB+EB+NO). Mismas 152 neuronas y mismo eb_angles.csv que `malecns`.
"""
import os
import shutil

import pandas as pd

from .evaluate import INTERIM_DIR
from .extract_graph import RAW_DIR

SRC = os.path.join(RAW_DIR, "malecns")
VARIANTS = {"malecns_no_NO": lambda roi: roi != "NO",
            "malecns_pb_eb_no": lambda roi: roi in ("PB", "EB", "NO")}


def main():
    conn = pd.read_csv(os.path.join(INTERIM_DIR, "adjacency_by_roi.csv"))
    for name, keep in VARIANTS.items():
        out = os.path.join(RAW_DIR, name)
        os.makedirs(out, exist_ok=True)
        sub = conn[conn["roi"].map(keep)]
        e = sub.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
        e.columns = ["source", "target", "weight"]
        e.to_csv(os.path.join(out, "graph_no_sign.csv"), index=False)
        for f in ("nodes.csv", "eb_angles.csv"):
            shutil.copy(os.path.join(SRC, f), os.path.join(out, f))
        print(name, len(e), "aristas; peso", int(e.weight.sum()), "de", int(conn.weight.sum()))


if __name__ == "__main__":
    main()

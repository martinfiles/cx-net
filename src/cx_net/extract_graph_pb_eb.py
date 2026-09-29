"""
Grafo restringido a los dos compartimentos del mecanismo de rotación rápido
(PB, EB), como complemento del grafo por defecto (`extract_graph.py`), que
suma sinapsis de TODAS las regiones donde coexisten los 152 nodos del núcleo
(incluye NO, FB, LAL(L/R), CentralBrain-unspecified: ver
`pen_offset_by_region.py` y docs/lab-notebook.md, 2026-09-28 (2)).

Reutiliza las MISMAS 152 neuronas y el mismo `eb_angles.csv` que
`data/raw/malecns/` (la fase de cada neurona no depende de qué aristas se
entrenan), y solo recalcula `graph_no_sign.csv` filtrando `roi` a {PB, EB}
antes de sumar pesos por par (source, target).

Salida: data/raw/malecns_pb_eb/{nodes.csv, graph_no_sign.csv, eb_angles.csv}
(mismo esquema que data/raw/malecns/, para que `graph_utils.load_cx_graph`
funcione sin cambios pasando `data_dir=".../malecns_pb_eb"`).
"""
import os
import shutil

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, fetch_adjacencies

from .extract_graph import RAW_DIR, SERVER

SRC_DIR = os.path.join(RAW_DIR, "malecns")
OUT_DIR = os.path.join(RAW_DIR, "malecns_pb_eb")
COMPARTMENTS = {"PB", "EB"}


def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    client = Client(SERVER, dataset="male-cns:v1.0", token=os.environ["NEUPRINT_TOKEN"])

    os.makedirs(OUT_DIR, exist_ok=True)
    nodes = pd.read_csv(os.path.join(SRC_DIR, "nodes.csv"))
    ids = nodes["bodyId"].tolist()

    print(f"Consultando adyacencias de {len(ids)} neuronas con desglose por roi...")
    _, conn = fetch_adjacencies(sources=ids, targets=ids, client=client)
    before = conn["weight"].sum()
    conn = conn[conn["roi"].isin(COMPARTMENTS)]
    after = conn["weight"].sum()
    print(f"Peso total: {before} (todas las regiones) -> {after} (solo PB+EB), "
          f"{after / before:.1%} conservado")

    edges = conn.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
    edges.columns = ["source", "target", "weight"]
    edges.to_csv(os.path.join(OUT_DIR, "graph_no_sign.csv"), index=False)
    print(f"{len(edges)} aristas (antes {sum(1 for _ in open(os.path.join(SRC_DIR, 'graph_no_sign.csv'))) - 1})")

    nodes.to_csv(os.path.join(OUT_DIR, "nodes.csv"), index=False)
    shutil.copy(os.path.join(SRC_DIR, "eb_angles.csv"), os.path.join(OUT_DIR, "eb_angles.csv"))
    print(f"Copiados nodes.csv y eb_angles.csv (sin cambios) a {OUT_DIR}")


if __name__ == "__main__":
    main()

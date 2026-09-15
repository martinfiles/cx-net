"""
Fase 1 — Extracción del grafo del complejo central (CX).

Descarga la conectividad de los neuropilos PB / EB / FB / NO desde el
hemibrain (piloto, API de neuprint madura) y la exporta en dos archivos
separados:

  - graph_no_sign.csv   -> topología pura (source, target, weight), SIN signo.
                           Esto es lo único que verá el modelo durante el
                           entrenamiento (fase 2).
  - ground_truth_nt.csv -> neurotransmisor predicho por neurona (bodyId, type,
                           predicted_nt). Se guarda aparte y NO se usa hasta
                           la fase de evaluación (fase 4).

Requiere NEUPRINT_TOKEN en el entorno (ver .env.example). Antes de la
primera ejecución real: confirmar contra la documentación viva de
neuprint-python el nombre exacto de la columna de neurotransmisor
predicho para el dataset/version en uso (ha cambiado de nombre entre
versiones del hemibrain) -- está marcado más abajo con TODO.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

CX_ROIS = ["PB", "EB", "FB", "NO"]
DATASET = "hemibrain:v1.2.1"
SERVER = "neuprint.janelia.org"

# Núcleo del sistema de dirección de cabeza (ring attractor), no todo el CX:
# EPG (compás), PEN_a/PEN_b (integran velocidad angular), PEG (cierra el bucle
# FB-EB-PB), Delta7 (inhibición lateral que mantiene un único "bump").
# Nombres verificados en vivo contra hemibrain:v1.2.1 (ver docs/lab-notebook.md,
# entrada 2026-09-16). Deja fuera a los ~30 subtipos de ring neurons (ER/ExR,
# entrada visual) y a los PFN/PFL/hDelta/vDelta del fan-shaped body: son parte
# del CX completo pero no del núcleo de heading, y multiplicarían el tamaño
# del grafo sin aportar a H1/H2 en esta primera iteración.
CORE_HEAD_DIRECTION_TYPES = [
    "EPG",
    "EPGt",
    "PEN_a(PEN1)",
    "PEN_b(PEN2)",
    "PEG",
    "Delta7",
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def get_client() -> Client:
    load_dotenv()
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "Falta NEUPRINT_TOKEN. Copia .env.example a .env y añade tu token "
            "de https://neuprint.janelia.org (Account > Auth Token)."
        )
    return Client(SERVER, dataset=DATASET, token=token)


def fetch_cx_neurons(client: Client) -> pd.DataFrame:
    """Neuronas 'Traced' del núcleo del sistema de dirección de cabeza.

    roi_req="any" es imprescindible: por defecto NeuronCriteria exige
    presencia en TODOS los rois listados a la vez, lo que deja fuera a casi
    todas las neuronas del circuito (solo 30 de varios cientos esperadas).

    El filtro por rois por sí solo es demasiado laxo (3.085 neuronas, incluye
    fibras de paso sin relación con el circuito de rumbo) -- se acota además
    por tipo celular canónico (CORE_HEAD_DIRECTION_TYPES).
    """
    criteria = NC(rois=CX_ROIS, roi_req="any", status="Traced")
    neuron_df, _roi_counts_df = fetch_neurons(criteria, client=client)
    return neuron_df[neuron_df["type"].isin(CORE_HEAD_DIRECTION_TYPES)].reset_index(drop=True)


def fetch_cx_connectivity(client: Client, neuron_df: pd.DataFrame) -> pd.DataFrame:
    """Conectividad neurona-a-neurona restringida al propio subconjunto CX.

    fetch_adjacencies devuelve (neurons_df, roi_conn_df) EN ESE ORDEN -- el
    primer valor es una tabla de neuronas, no de conexiones. Confirmado
    empíricamente con help(fetch_adjacencies) tras un primer intento fallido
    que asumía el orden contrario.
    """
    body_ids = neuron_df["bodyId"].tolist()
    _touched_neurons_df, conn_df = fetch_adjacencies(
        sources=body_ids, targets=body_ids, client=client
    )
    return conn_df


def build_topology_only_graph(conn_df: pd.DataFrame) -> pd.DataFrame:
    """(source, target, weight) sin ninguna información de signo/neurotransmisor."""
    if conn_df.empty:
        raise ValueError(
            "fetch_adjacencies devolvió 0 conexiones. Revisa el criterio de "
            "selección de neuronas (fetch_cx_neurons) antes de continuar."
        )
    edges = conn_df.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
    edges.columns = ["source", "target", "weight"]
    return edges


def build_ground_truth_nt(neuron_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Tabla oculta de neurotransmisor real por neurona. Se guarda aparte y no
    se toca hasta la evaluación (fase 4).

    Confirmado en vivo (2026-09-16): el hemibrain (hemibrain:v1.2.1) NO
    incluye neurotransmisor predicho entre las propiedades de Neuron en
    neuprint -- esa anotación (Eckstein et al.) solo está integrada en el
    MaleCNS v1.0 (2026), vía CAVE. Por tanto esta función devuelve None en
    el piloto sobre hemibrain; el ground truth real se extraerá en la
    migración a MaleCNS (ver docs/lab-notebook.md).
    """
    nt_column_candidates = ["predictedNt", "consensusNt", "celltypePredictedNt"]
    nt_column = next((c for c in nt_column_candidates if c in neuron_df.columns), None)
    if nt_column is None:
        return None
    return neuron_df[["bodyId", "type", nt_column]].rename(
        columns={nt_column: "predicted_nt"}
    )


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    client = get_client()

    print(f"Consultando neuronas del CX ({', '.join(CX_ROIS)}) en {DATASET}...")
    neuron_df = fetch_cx_neurons(client)
    print(f"  {len(neuron_df)} neuronas encontradas.")

    print("Consultando conectividad interna del subcircuito...")
    conn_df = fetch_cx_connectivity(client, neuron_df)

    topology = build_topology_only_graph(conn_df)
    ground_truth = build_ground_truth_nt(neuron_df)

    topology_path = os.path.join(RAW_DIR, "graph_no_sign.csv")
    topology.to_csv(topology_path, index=False)
    print(f"Grafo sin signo guardado en {topology_path} ({len(topology)} aristas, {len(neuron_df)} neuronas).")

    nodes_path = os.path.join(RAW_DIR, "nodes.csv")
    neuron_df[["bodyId", "type", "instance"]].to_csv(nodes_path, index=False)
    print(f"Metadatos de nodos guardados en {nodes_path} (necesarios para la Fase 2: lado e instancia anatómica).")

    if ground_truth is None:
        print(
            "Aviso: este dataset (hemibrain) no incluye neurotransmisor predicho. "
            "Ground truth pendiente de MaleCNS v1.0 (fase de migración a CAVE)."
        )
    else:
        gt_path = os.path.join(RAW_DIR, "ground_truth_nt.csv")
        ground_truth.to_csv(gt_path, index=False)
        print(f"Ground truth de neurotransmisor guardado en {gt_path} (uso restringido a evaluación).")


if __name__ == "__main__":
    main()

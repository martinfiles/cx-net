"""
Formaliza y verifica la simetría de signo propuesta por la revisión externa
(2026-09-28) para la enumeración de 64 patrones (`analyze_types_3seeds.py`).

Afirmación del revisor: con una activación impar (tanh) y sesgos entrenables,
la dinámica r_{t+1} = f(W r_t + I_t + b) es invariante ante r -> Dr, W -> DWD,
b -> Db, I -> DI, para cualquier D diagonal con entradas ±1 -- el "switching"
de grafos con signo. Predicción: solo el signo neto de cada CICLO del grafo
sería identificable, no el signo de cada arista o tipo por separado.

Esa afirmación es correcta para un modelo con GANANCIAS LIBRES POR ARISTA (o
incluso por par de tipos, pero de signo libre). Pero el modelo de este
proyecto (`model.py`, `learn_type_params=True`) tiene:
  - un signo por TIPO (6 parámetros, no 9.160 por arista);
  - una ganancia por PAR DE TIPOS (36 parámetros), constreñida positiva
    (exp(log_pair_gain)).
Bajo esa restricción de positividad, D=diag(d_1..d_152) con d_i = D_type(type(i))
(el mismo signo para toda neurona del mismo tipo, que es como se aplica la
máscara) es una simetría del modelo tal como está parametrizado SOLO SI, para
todo par de tipos (a, b) con al menos una arista a->b, D_type(a) * D_type(b) = +1
(si no, la ganancia compensatoria tendría que ser negativa, y el modelo la
prohíbe). Con un grafo de tipos COMPLETO (los 36 pares dirigidos, incluidos los
6 lazos consigo mismo, tienen peso > 0 -- se verifica abajo, tanto para el
grafo mezclado como para el grafo PB+EB), esa condición para TODOS los pares
fuerza D_type(a) = D_type(b) para cualquier par de tipos: el único D
compatible es el global (D_type ≡ +1 o D_type ≡ -1, que además cambiaría el
signo de la entrada de velocidad si no se ajusta a la vez, así que ni siquiera
ese es gratis).

**Conclusión (corrige la formulación informal de `docs/preprint-discussion.md`,
sección 6): con la parametrización real de este modelo, NO existe una simetría
EXACTA de signo por tipo no trivial.** Pero la 3ª revisión (2026-09-29) señala
que de ahí no se sigue que las diferencias sean «dinámicas reales»: el grafo de
tipos es completo solo trivialmente (p. ej. PEN→Delta7 son 12-14 aristas y 14-15
sinapsis en total) y varios tipos (PEG, EPGt) tienen acoplamiento de salida muy
débil, así que existen simetrías APROXIMADAS; y parte de la variación es ruido de
optimización (16 de 63 patrones no son unánimes entre semillas). Ver
`pair_strength()` (matrices 6x6 y ganancias entrenadas) y
`results/sign_symmetry_pairs.json`. Esto no explica por qué la tabla Delta7×EPG×PEN_b (sección 6) separa
tan bien los patrones que integran -- esa es una regularidad empírica sobre la
dinámica, no una consecuencia de la simetría formal.

Salida: imprime la verificación y las condiciones de compatibilidad D_a*D_b=+1
por par de tipos, para los dos grafos (mezclado y PB+EB).
"""
import itertools
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def type_adjacency(data_dir):
    nodes = pd.read_csv(os.path.join(data_dir, "nodes.csv"))
    edges = pd.read_csv(os.path.join(data_dir, "graph_no_sign.csv"))
    type_of = dict(zip(nodes["bodyId"], nodes["type"]))
    edges["pre_type"] = edges["source"].map(type_of)
    edges["post_type"] = edges["target"].map(type_of)
    types = sorted(nodes["type"].unique())
    mat = (edges.groupby(["pre_type", "post_type"])["weight"].sum()
           .unstack(fill_value=0).reindex(index=types, columns=types, fill_value=0))
    return types, mat


def check_symmetry(label, data_dir):
    types, mat = type_adjacency(data_dir)
    present = (mat.to_numpy() > 0)
    complete = present.all()
    print(f"\n== {label} ({data_dir}) ==")
    print(f"tipos: {types}")
    print(f"grafo de tipos completo (36/36 pares con peso > 0): {complete}")
    if not complete:
        missing = [(a, b) for a in types for b in types if mat.loc[a, b] == 0]
        print(f"pares SIN arista (posible grado de libertad para D no trivial): {missing}")
        return

    # D compatible con positividad de ganancias: D_a * D_b = +1 para TODO par
    # (a,b) con arista -> como el grafo es completo, D debe ser constante.
    # Verificación por fuerza bruta sobre los 2^6 vectores D (constatar que
    # solo los 2 triviales son compatibles).
    compatible = []
    for bits in itertools.product([1, -1], repeat=len(types)):
        d = dict(zip(types, bits))
        ok = all(d[a] * d[b] == 1 for a in types for b in types if mat.loc[a, b] > 0)
        if ok:
            compatible.append(bits)
    print(f"vectores D compatibles con ganancias positivas (de 2^6=64 posibles): {len(compatible)}")
    for bits in compatible:
        print(f"  {dict(zip(types, bits))}")
    print("-> ningún D no trivial es una simetría del modelo tal como está parametrizado."
          if len(compatible) == 2 else "-> AVISO: hay D no triviales, revisar.")


if __name__ == "__main__":
    check_symmetry("grafo mezclado (todas las regiones)", os.path.join(RAW_DIR, "malecns"))
    check_symmetry("grafo PB+EB", os.path.join(RAW_DIR, "malecns_pb_eb"))


def pair_strength(data_dir=os.path.join(RAW_DIR, "malecns")):
    """Respuesta a la 3ª revisión (punto C): "no hay simetría exacta" no implica
    "las diferencias son reales" si algún par de tipos es casi nulo (aunque sea
    por sinapsis escasas o porque la ganancia entrenada lo lleva al mínimo
    exp(-5) = 0.0067). Muestra, por par origen->destino, la fracción de la
    entrada total de la neurona destino (media ponderada) y la ganancia entrenada
    de las 3 semillas reales (`results_{realctl_tanh_p,dose_real_seed1_p,dose_real_seed2_p}`),
    y el acoplamiento efectivo = fracción x exp(ganancia). Escribe
    results/sign_symmetry_pairs.json."""
    import json
    import numpy as np
    from .evaluate import INTERIM_DIR
    nodes = pd.read_csv(os.path.join(data_dir, "nodes.csv"))
    edges = pd.read_csv(os.path.join(data_dir, "graph_no_sign.csv"))
    type_of = dict(zip(nodes["bodyId"], nodes["type"]))
    edges["pre"] = edges["source"].map(type_of)
    edges["post"] = edges["target"].map(type_of)
    edges["norm"] = edges["weight"] / edges.groupby("target")["weight"].transform("sum").clip(lower=1.0)
    names = [t for t in pd.factorize(nodes["type"])[1]]  # orden de type_ids en train.py
    mat_norm = edges.groupby(["pre", "post"])["norm"].sum().unstack(fill_value=0).reindex(index=names, columns=names, fill_value=0)
    n_post = nodes["type"].value_counts()
    mat_norm = mat_norm / n_post.reindex(names).to_numpy()[None, :]  # entrada normalizada media por neurona destino
    gains = []
    for lab in ["realctl_tanh_p", "dose_real_seed1_p", "dose_real_seed2_p"]:
        r = json.load(open(os.path.join(INTERIM_DIR, f"results_{lab}.json")))["results"]
        gains.append(np.exp(np.clip(np.array(r["log_pair_gain"]).reshape(len(names), len(names)), -5, 5)))
    gains = np.array(gains)
    eff = mat_norm.to_numpy()[None] * gains * 2.0  # recurrent_gain = 2
    out = {"types": names, "input_fraction_mean_per_target_neuron": mat_norm.round(4).values.tolist(),
           "trained_gain_by_seed": gains.round(4).tolist(), "effective_coupling_by_seed": eff.round(4).tolist(),
           "n_pairs_gain_at_floor_by_seed": [int((g < 0.0068).sum()) for g in gains]}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "sign_symmetry_pairs.json"), "w"), indent=1)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("tipos:", names)
    print("fracción de entrada (media por neurona destino), filas=origen, cols=destino:\n", mat_norm.to_numpy())
    for s in range(3):
        print(f"ganancia entrenada semilla {s}:\n", gains[s])
        print(f"acoplamiento efectivo semilla {s}:\n", eff[s])
    print("pares con ganancia en el mínimo:", out["n_pairs_gain_at_floor_by_seed"])
    # variación del held-out al cambiar SOLO el signo de EPGt y PEG (bits 4 y 8) frente al rango global
    ta = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "types_analysis_3seeds.json")))
    h = {x["mask"]: x["mean_held_out"] for x in ta}
    ig = {x["mask"]: x["integra_majority"] for x in ta}
    groups = {}
    for m in range(64):
        groups.setdefault(m & ~(4 | 8), []).append(m)
    spreads = [max(h[m] for m in g) - min(h[m] for m in g) for g in groups.values()]
    flips = sum(len({ig[m] for m in g}) > 1 for g in groups.values())
    out["epgt_peg_sign_only"] = {"n_groups": len(groups), "median_heldout_range_within_group": float(np.median(spreads)),
                                 "max_heldout_range_within_group": float(max(spreads)),
                                 "groups_where_majority_verdict_changes": int(flips),
                                 "global_heldout_sd": float(np.std(list(h.values())))}
    print("solo cambia signo EPGt/PEG:", out["epgt_peg_sign_only"])
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "sign_symmetry_pairs.json"), "w"), indent=1)

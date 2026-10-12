"""
Agrega el experimento del aprendiz de signos sobre la tarea OU corregida (preregistrado en el cuaderno,
entrada 2026-10-12 (3)) a partir de los `results_signlearn_*.json` de `data/interim/` (no reentrena nada).

Brazos: A (6 signos por tipo, inicio neutro), B (6 signos por tipo, inicio binario aleatorio),
C (un signo por arista), D (6 signos por tipo, inicio en la asignacion real).

Criterio de «integra» (el de la enumeracion): perdida incremental held-out < umbral «recordar theta0» y pendiente > 0.5,
tanto con el modelo continuo como con los signos binarizados (+-1 duro). «Recupera la real» = patron aprendido
(mascara por tipo) == 1 (solo Delta7 inhibe). Referencia de azar: 1/37 (el aprendiz elige uno al azar entre los 37
patrones que integran en la enumeracion).

Salida: results/sign_learner_v3.json
"""
import glob
import json
import math
import os
import re
from collections import Counter

from .evaluate import INTERIM_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SHORT = ["D7", "EPG", "EPGt", "PEG", "PENa", "PENb"]
ARMS = {"A": "6 signos por tipo, inicio neutro", "B": "6 signos por tipo, inicio binario aleatorio",
        "C": "un signo por arista, inicio neutro", "D": "6 signos por tipo, inicio en la real"}
CHANCE = 1 / 37


def binom_tail(k, n, p):
    """P(X >= k) para X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def mask_names(mask):
    return "+".join(SHORT[i] for i in range(6) if (mask >> i) & 1) or "ninguno"


def load_runs(memory):
    runs = []
    for path in sorted(glob.glob(os.path.join(INTERIM_DIR, "results_signlearn_*_s*.json"))):
        m = re.search(r"signlearn_([A-D])_s(\d+)", path)
        r = json.load(open(path))["results"]
        d, la = r["diagnostics"], r["learned_sign_analysis"]
        b = la["binarized"]["diagnostics"]
        run = {"arm": m.group(1), "seed": int(m.group(2)),
               "increment_loss": d["increment_loss"], "slope": d["slope"],
               "integra": bool(d["increment_loss"] < memory and d["slope"] > 0.5),
               "bin_increment_loss": b["increment_loss"], "bin_slope": b["slope"],
               "bin_integra": bool(b["increment_loss"] < memory and b["slope"] > 0.5),
               "agreement_with_real_edge_sign": la["agreement_with_real_edge_sign"],
               "frac_inhibitory_by_source_type": la["frac_inhibitory_by_source_type"]}
        if "type_sign_mask" in la:
            run["mask"] = la["type_sign_mask"]
            run["type_sign_param"] = la["type_sign_param"]
        runs.append(run)
    return runs


def main():
    ref = json.load(open(os.path.join(RESULTS_DIR, "task_v3_reference.json")))
    memory = ref["ou_nueva"]["memorize_theta0"]["increment"]
    enum = {x["mask"]: x for x in json.load(open(os.path.join(RESULTS_DIR, "types_analysis_v3.json")))["patterns"]}
    runs = load_runs(memory)
    out = {"memory_threshold": memory, "chance_real": CHANCE, "arms": {}}
    for arm, desc in ARMS.items():
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs)
        a = {"description": desc, "n_runs": n, "n_integra": sum(r["integra"] for r in rs),
             "n_bin_integra": sum(r["bin_integra"] for r in rs),
             "mean_agreement_with_real_edge_sign": sum(r["agreement_with_real_edge_sign"] for r in rs) / n,
             "runs": rs}
        if arm != "C":
            masks = Counter(r["mask"] for r in rs)
            n_real = masks.get(1, 0)
            a.update({
                "mask_counts": {f"{m} ({mask_names(m)})": c for m, c in masks.most_common()},
                "n_real_mask": n_real,
                "n_d7_inhibitory": sum(bool(r["mask"] & 1) for r in rs),
                "n_mask_integrates_in_enumeration": sum(enum[r["mask"]]["integra_majority"] for r in rs),
                "p_real_at_least_k_under_chance": binom_tail(n_real, n, CHANCE),
                "n_real_mask_and_bin_integra": sum(r["mask"] == 1 and r["bin_integra"] for r in rs),
            })
        out["arms"][arm] = a
        line = (f"[{arm}] {desc}: n={n}; integran (continuo / binarizado): {a['n_integra']} / {a['n_bin_integra']}; "
                f"acuerdo medio con el signo real por arista: {a['mean_agreement_with_real_edge_sign']:.3f}")
        if arm != "C":
            line += (f"; patron == real: {a['n_real_mask']}/{n}; Delta7 inhibidor: {a['n_d7_inhibitory']}/{n}; "
                     f"patrones: {a['mask_counts']}")
        print(line)
    json.dump(out, open(os.path.join(RESULTS_DIR, "sign_learner_v3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

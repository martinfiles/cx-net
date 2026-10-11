"""
Agrega la enumeración de los 64 patrones de signo por tipo sobre la tarea OU
corregida (entrada 2026-10-09 del cuaderno; `data/interim/launch_types_v3.sh`)
a partir de los `results_*.json` de `data/interim/` (no reentrena nada).

Convención de nombres de corrida:
  - máscara 1 (asignación real): "taskv3_real_seed{s}_p".
  - resto de máscaras m: "taskv3_mask{m}_s{s}_p".

Criterio (fijado antes de ver resultados): «integra» = pérdida incremental en
held-out < «recordar theta0» en esta tarea (`results/task_v3_reference.json`) Y
pendiente > 0.5; por patrón, voto de mayoría entre semillas. Solo se usan las
semillas que estén completas para los 64 patrones.

Salida: results/types_analysis_v3.json
"""
import json
import os

from .evaluate import INTERIM_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SHORT = ["D7", "EPG", "EPGt", "PEG", "PENa", "PENb"]
SEEDS = (0, 1, 2)


def label_for(mask, seed):
    return f"taskv3_real_seed{seed}_p" if mask == 1 else f"taskv3_mask{mask}_s{seed}_p"


def result_path(mask, seed):
    return os.path.join(INTERIM_DIR, f"results_{label_for(mask, seed)}.json")


def load_run(mask, seed, memory):
    r = json.load(open(result_path(mask, seed)))["results"]
    d = r["diagnostics"]
    return {"seed": seed, "increment_loss": d["increment_loss"], "held_out": r["held_out_loss"],
            "slope": d["slope"], "anchor_err_deg": d["anchor_err_deg"],
            "integra": bool(d["increment_loss"] < memory and d["slope"] > 0.5)}


def load_pattern(mask, seeds, memory):
    runs = [load_run(mask, s, memory) for s in seeds]
    n_integra = sum(x["integra"] for x in runs)
    return {
        "mask": mask,
        "inhibitory": [SHORT[i] for i in range(6) if (mask >> i) & 1],
        "n_seeds": len(runs),
        "n_integra": n_integra,
        "mean_increment_loss": sum(x["increment_loss"] for x in runs) / len(runs),
        "mean_slope": sum(x["slope"] for x in runs) / len(runs),
        "runs": runs,
        "integra_majority": 2 * n_integra > len(runs),
        "integra_all": n_integra == len(runs),
    }


def main():
    ref = json.load(open(os.path.join(RESULTS_DIR, "task_v3_reference.json")))
    memory = ref["ou_nueva"]["memorize_theta0"]["increment"]
    seeds = [s for s in SEEDS if all(os.path.exists(result_path(m, s)) for m in range(64))]
    if not seeds:
        raise SystemExit("ninguna semilla completa para los 64 patrones")
    res = [load_pattern(m, seeds, memory) for m in range(64)]
    real = res[1]
    ranked = sorted(res, key=lambda x: x["mean_increment_loss"])
    rank = next(i for i, x in enumerate(ranked, start=1) if x["mask"] == 1)
    n_maj = sum(x["integra_majority"] for x in res)
    n_all = sum(x["integra_all"] for x in res)

    by_type = {}
    for i, t in enumerate(SHORT):
        exc = [x for x in res if not (x["mask"] >> i) & 1]
        inh = [x for x in res if (x["mask"] >> i) & 1]
        by_type[t] = {"integra_exc": sum(x["integra_majority"] for x in exc),
                      "integra_inh": sum(x["integra_majority"] for x in inh), "n_each": 32}

    # concordancia con el veredicto de la tarea antigua (3 semillas, voto de mayoría)
    old = {x["mask"]: x["integra_majority"]
           for x in json.load(open(os.path.join(RESULTS_DIR, "types_analysis_3seeds.json")))}
    table = {"ambas": 0, "solo_antigua": 0, "solo_nueva": 0, "ninguna": 0}
    for x in res:
        key = {(True, True): "ambas", (True, False): "solo_antigua",
               (False, True): "solo_nueva", (False, False): "ninguna"}[(old[x["mask"]], x["integra_majority"])]
        table[key] += 1

    print(f"semillas completas: {seeds}; umbral «recordar theta0» (incremental): {memory:.3f}")
    print(f"REAL (máscara 1): pérdida incremental media {real['mean_increment_loss']:.3f}, pendiente "
          f"{real['mean_slope']:.2f}, integra en {real['n_integra']}/{len(seeds)}; puesto {rank}/64 "
          f"(p = {rank}/64 = {rank/64:.3f})")
    print(f"integran: por mayoría {n_maj}/64 | en todas las semillas {n_all}/64")
    print("mejores 8:", [(x["mask"], "+".join(x["inhibitory"]) or "ninguno", round(x["mean_increment_loss"], 3),
                          round(x["mean_slope"], 2)) for x in ranked[:8]])
    print("por tipo (integran con el tipo excitador / inhibidor, de 32):",
          {t: (v["integra_exc"], v["integra_inh"]) for t, v in by_type.items()})
    print("concordancia con la tarea antigua:", table)

    out = {"seeds": seeds, "memory_threshold": memory, "real_rank": rank, "n_integra_majority": n_maj,
           "n_integra_all": n_all, "by_type": by_type, "agreement_with_old_task": table, "patterns": res}
    json.dump(out, open(os.path.join(RESULTS_DIR, "types_analysis_v3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

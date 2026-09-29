"""
Agrega las 3 semillas de la enumeración de 64 patrones de signo por tipo
(entradas (12) y 2026-09-22 (2) del cuaderno) y reproduce
`results/types_analysis_3seeds.json` y `results/types_explorer_3seeds.json`
a partir de los `results_*.json` ya presentes en `data/interim/` (no
reentrena nada).

Convención de nombres de corrida (`launch_types.sh`, `launch_seeds.sh`):
  - máscara 1 (solo Delta7 inhibe = asignación real): "realctl_tanh_p" (seed 0),
    "dose_real_seed1_p" / "dose_real_seed2_p" (seeds 1/2).
  - resto de máscaras m: "types_mask{m}_p" (seed 0), "types_mask{m}_s1_p" /
    "types_mask{m}_s2_p" (seeds 1/2).

Salida: results/types_analysis_3seeds.json, results/types_explorer_3seeds.json
"""
import json
import os

from .evaluate import INTERIM_DIR
from .train import TYPE_ORDER

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
MEMORY = 0.466
SHORT = ["D7", "EPG", "EPGt", "PEG", "PENa", "PENb"]

REAL_MASK_LABELS = {0: "realctl_tanh_p", 1: "dose_real_seed1_p", 2: "dose_real_seed2_p"}


def label_for(mask, seed):
    if mask == 1:
        return REAL_MASK_LABELS[seed]
    return f"types_mask{mask}_p" if seed == 0 else f"types_mask{mask}_s{seed}_p"


def load_run(mask, seed):
    label = label_for(mask, seed)
    r = json.load(open(os.path.join(INTERIM_DIR, f"results_{label}.json")))["results"]
    d = r["diagnostics"]
    held_out, slope = r["held_out_loss"], d["slope"]
    return {"seed": seed, "held_out": held_out, "slope": slope,
            "anchor_err_deg": d["anchor_err_deg"],
            "integra": bool(held_out < MEMORY and slope > 0.5)}


def load_pattern(mask):
    runs = [load_run(mask, s) for s in (0, 1, 2)]
    n_integra = sum(x["integra"] for x in runs)
    mean_held_out = sum(x["held_out"] for x in runs) / len(runs)
    return {
        "mask": mask,
        "inhibitory": [SHORT[i] for i in range(6) if (mask >> i) & 1],
        "n_seeds": len(runs),
        "n_integra": n_integra,
        "mean_held_out": mean_held_out,
        "runs": runs,
        "integra_majority": n_integra >= 2,
        "integra_any": n_integra >= 1,
        "integra_all": n_integra == 3,
    }


def main():
    res = [load_pattern(m) for m in range(64)]
    real = res[1]
    ranked = sorted(res, key=lambda x: x["mean_held_out"])
    rank = next(i for i, x in enumerate(ranked, start=1) if x["mask"] == 1)
    n_maj = sum(x["integra_majority"] for x in res)
    n_all = sum(x["integra_all"] for x in res)
    n_any = sum(x["integra_any"] for x in res)
    not_unanimous = [x["mask"] for x in res if 0 < x["n_integra"] < 3]
    flips = [(x["mask"], x["runs"][0]["integra"], x["integra_majority"])
             for x in res if x["runs"][0]["integra"] != x["integra_majority"]]

    print(f"REAL (máscara 1): held-out medio {real['mean_held_out']:.3f}, integra en "
          f"{real['n_integra']}/3 semillas, puesto {rank}/64 por held-out medio "
          f"(p = {rank}/64 = {rank/64:.3f})")
    print(f"integran: en las 3 semillas {n_all}/64 | por mayoría {n_maj}/64 | "
          f"en alguna {n_any}/64")
    print(f"no unánimes entre semillas: {len(not_unanimous)}/63 (excluida la real): "
          f"{not_unanimous}")
    print(f"de esos, cambian el voto de mayoría respecto a la semilla 0: {len(flips)}: "
          f"{flips}")

    json.dump(res, open(os.path.join(RESULTS_DIR, "types_analysis_3seeds.json"), "w"),
               indent=1)

    # Formato compacto que consume docs/explorer.html: mask, held-out medio,
    # pendiente media, error de anclaje medio (grados).
    explorer = [{
        "m": x["mask"],
        "h": round(x["mean_held_out"], 4),
        "s": round(sum(r["slope"] for r in x["runs"]) / len(x["runs"]), 3),
        "a": round(sum(r["anchor_err_deg"] for r in x["runs"]) / len(x["runs"]), 1),
    } for x in res]
    json.dump(explorer, open(os.path.join(RESULTS_DIR, "types_explorer_3seeds.json"), "w"))


if __name__ == "__main__":
    main()

"""Análisis preespecificado de la enumeración de 64 patrones de signo por tipo (entrada (12))."""
import json
import os

import numpy as np

from .evaluate import INTERIM_DIR
from .train import TYPE_ORDER

MEMORY = 0.466
SHORT = ["D7", "EPG", "EPGt", "PEG", "PENa", "PENb"]


def load(mask):
    label = "realctl_tanh_p" if mask == 1 else f"types_mask{mask}_p"  # máscara 1 = real (seed 0)
    r = json.load(open(os.path.join(INTERIM_DIR, f"results_{label}.json")))["results"]
    d = r["diagnostics"]
    return dict(mask=mask, held_out=r["held_out_loss"], slope=d["slope"], anchor=d["anchor_err_deg"],
                inhibitory=[SHORT[i] for i in range(6) if (mask >> i) & 1])


def main():
    res = [load(m) for m in range(64)]
    for x in res:
        x["integra"] = bool(x["held_out"] < MEMORY and x["slope"] > 0.5)
    real = res[1]
    n_int = sum(x["integra"] for x in res)
    n_le = sum(x["held_out"] <= real["held_out"] for x in res)
    print(f"REAL (máscara 1, solo D7 inhibe): held-out {real['held_out']:.3f} pendiente {real['slope']:.2f}")
    print(f"patrones que integran: {n_int}/64 | p exacto (rango de la real: patrones con held-out <= real / 64) = {n_le}/64 = {n_le/64:.3f}\n")
    print("PATRONES QUE INTEGRAN (ordenados por held-out):")
    for x in sorted((x for x in res if x["integra"]), key=lambda x: x["held_out"]):
        print(f"  máscara {x['mask']:2d}: held-out {x['held_out']:.3f} pendiente {x['slope']:.2f} | inhiben: {','.join(x['inhibitory']) or '(ninguno)'}")
    print("\nFRACCIÓN DE PATRONES QUE INTEGRAN según el signo de cada tipo:")
    for i, t in enumerate(SHORT):
        exc = [x["integra"] for x in res if not (x["mask"] >> i) & 1]
        inh = [x["integra"] for x in res if (x["mask"] >> i) & 1]
        print(f"  {t:5s}: excitador {sum(exc)}/{len(exc)}  inhibidor {sum(inh)}/{len(inh)}")
    print("\nDistribución held-out de los que NO integran: min %.3f mediana %.3f max %.3f" % tuple(
        np.percentile([x["held_out"] for x in res if not x["integra"]], [0, 50, 100])))
    json.dump(res, open(os.path.join(INTERIM_DIR, "types_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

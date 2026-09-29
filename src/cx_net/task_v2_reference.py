"""
Referencias de pérdida para la tarea rediseñada (2026-09-28, respuesta a la
revisión externa; ver `task.py`, `velocity_process="ou"`): decodificador
constante, "recordar theta0 sin integrar", integrador perfecto y una rejilla
de integradores con ganancia constante != 1, para mostrar que el hueco
señalado por el revisor ("un integrador con ganancia 0.54 pasaba el criterio
de integra con pérdida 0.138") queda cerrado con la tarea nueva.

Mismos hiperparámetros de tarea que las corridas de validación
(`data/interim/launch_taskv2.sh`): T=300, max_av=0.15, hold_prob=0.3,
velocity_process="ou", tau_v=30.

Salida: results/task_v2_reference.json
"""
import json
import os

import numpy as np
import torch

from .task import circular_loss, generate_held_out_set

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
TASK_KW = dict(n_trials=30, T=300, hold_prob=0.3, anchor=True, max_av=0.15,
               velocity_process="ou", tau_v=30.0)
GAINS = [0.54, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20]


def main():
    trials = generate_held_out_set(**TASK_KW)
    median_laps = float(np.median([abs(h[-1] - t) for _, h, t in trials]) / (2 * np.pi))
    frac_over_one_lap = float(np.mean([abs(h[-1] - t) > 2 * np.pi for _, h, t in trials]))

    def loss_of(decoder_fn):
        losses = []
        for av, heading, theta0 in trials:
            dec = decoder_fn(av, heading, theta0)
            losses.append(circular_loss(torch.as_tensor(dec, dtype=torch.float32), heading).item())
        return float(np.mean(losses))

    out = {
        "task_kwargs": TASK_KW,
        "median_laps_per_trial": median_laps,
        "frac_trials_over_one_lap": frac_over_one_lap,
        "constant_decoder_0": loss_of(lambda av, h, t0: np.zeros_like(h)),
        "memorize_theta0_no_integration": loss_of(lambda av, h, t0: np.full_like(h, t0)),
        "perfect_integrator": loss_of(lambda av, h, t0: h),
        "constant_gain_integrator": {
            str(g): loss_of(lambda av, h, t0, g=g: t0 + g * np.cumsum(av)) for g in GAINS
        },
    }
    json.dump(out, open(os.path.join(RESULTS_DIR, "task_v2_reference.json"), "w"), indent=1)
    print(f"mediana de vueltas por ensayo: {median_laps:.2f} ({frac_over_one_lap:.0%} superan 1 vuelta)")
    print(f"constante: {out['constant_decoder_0']:.3f}  memoria: {out['memorize_theta0_no_integration']:.3f}  "
          f"perfecto: {out['perfect_integrator']:.4f}")
    for g, l in out["constant_gain_integrator"].items():
        print(f"  ganancia {g}: pérdida {l:.4f}")


if __name__ == "__main__":
    main()

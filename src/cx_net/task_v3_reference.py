"""
Referencias para la tarea OU con velocidad en el rango antiguo (3ª revisión
externa, punto A, 2026-09-29).

El revisor midió que la velocidad que "ve" la red (filtrada con su tau=10) en
la tarea OU anterior (max_av=0.15, tau_v=30) tiene media 0.078 y percentil 90
0.19, mientras que ninguna red entrenada movió el bump a más de ≈0.037-0.055
rad/paso (`results/bump_check.json`). Este script reproduce esa medida para
las tres tareas (vieja/blanca, OU anterior, OU nueva) y calcula las pérdidas
de referencia (circular e incremental) de integradores ideales de ganancia
constante. AVISO: estos integradores ideales (cumsum) NO son redes de esta
clase de modelos; solo acotan la escala de las pérdidas, no demuestran que la
solución exista en el espacio de búsqueda.

Salida: results/task_v3_reference.json
"""
import json
import os

import numpy as np
import torch

from .task import circular_loss, generate_held_out_set, increment_loss

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
TAU_NET = 10.0
TASKS = {
    "vieja_blanca": dict(T=200, hold_prob=0.3, anchor=True, max_av=0.15, velocity_process="white", tau_v=15.0),
    "ou_anterior": dict(T=300, hold_prob=0.3, anchor=True, max_av=0.15, velocity_process="ou", tau_v=30.0),
    "ou_nueva": dict(T=500, hold_prob=0.3, anchor=True, max_av=0.04, velocity_process="ou", tau_v=50.0),
}
GAINS = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3]


def filtered_speed(av, tau=TAU_NET):
    a = 1.0 / tau
    out = np.empty(len(av))
    x = 0.0
    for t, v in enumerate(av):
        x += a * (v - x)
        out[t] = x
    return np.abs(out)


def main():
    res = {}
    for name, kw in TASKS.items():
        trials = generate_held_out_set(n_trials=30, **kw)
        sp = np.concatenate([filtered_speed(av) for av, _, _ in trials])
        laps = [abs(h[-1] - t) / (2 * np.pi) for _, h, t in trials]
        full = np.mean([np.max(np.abs(h - t)) > 2 * np.pi for _, h, t in trials])

        def loss(fn, f):
            return float(np.mean([f(torch.as_tensor(fn(av, h, t), dtype=torch.float32), h).item()
                                  for av, h, t in trials]))
        res[name] = {
            "kwargs": kw,
            "filtered_speed_mean": float(sp.mean()), "filtered_speed_p90": float(np.percentile(sp, 90)),
            "frac_time_over_0.08": float((sp > 0.08).mean()),
            "median_net_laps": float(np.median(laps)), "frac_trials_reaching_full_lap": float(full),
            "memorize_theta0": {"circular": loss(lambda a, h, t: np.full_like(h, t), circular_loss),
                                "increment": loss(lambda a, h, t: np.full_like(h, t), increment_loss)},
            "gain_integrator": {str(g): {"circular": loss(lambda a, h, t, g=g: t + g * np.cumsum(a), circular_loss),
                                         "increment": loss(lambda a, h, t, g=g: t + g * np.cumsum(a), increment_loss)}
                                for g in GAINS},
        }
        r = res[name]
        print(f"{name:14s} vel. filtrada media {r['filtered_speed_mean']:.3f} p90 {r['filtered_speed_p90']:.3f} "
              f"t>0.08 {r['frac_time_over_0.08']:.0%} | vueltas netas med {r['median_net_laps']:.2f}, "
              f"{r['frac_trials_reaching_full_lap']:.0%} llegan a 1 vuelta | memoria circ {r['memorize_theta0']['circular']:.2f} "
              f"inc {r['memorize_theta0']['increment']:.2f}")
        print("   ganancia -> (circ, inc):", {g: (round(v['circular'], 2), round(v['increment'], 2))
                                             for g, v in r["gain_integrator"].items()})
    json.dump(res, open(os.path.join(RESULTS_DIR, "task_v3_reference.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

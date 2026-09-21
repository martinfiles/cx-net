"""
Perfil del bump, kymographs y memoria de fase arbitraria (respuesta a la revisión externa:
"que la pérdida sea baja no prueba que haya un bump localizado que se desplace").
Complementa `bump_check.py` (ganancia de velocidad y localización global). Para los mismos
cuatro modelos entrenados:
  - kymograph: actividad media de las EPG/EPGt en 12 contenedores de fase del anillo frente
    al tiempo, con pista de fase 0 en t < 20 y velocidad constante av in {-0.04, 0, +0.04}
    por paso desde t = 20.
  - perfil del bump: con av = 0 y t >= 40, la actividad de las neuronas de compás alineada
    con la fase decodificada en cada instante (12 contenedores relativos a la fase decodificada, tras restar la media poblacional de cada instante, porque las tasas tienen signo), su
    ancho a media altura sobre el mínimo del perfil (grados), el contraste (máx-mín)/media|r| y la
    fracción de neuronas saturadas (|r| > 0.95).
  - memoria de fase: pista con 8 fases distintas y av = 0; error circular entre la fase
    decodificada y la de la pista en t = 60 y t = 199, y tasa de deriva.
Salida: results/bump_profile.json. Requiere data/interim/model_*.pt (no versionado).
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .graph_utils import load_cx_graph
from .model import CXRingNetwork
from .task import build_external_input, decode_heading

MODELS = {"real (solo Delta7 inhibe)": "realctl_tanh_p", "solo EPG inhibe": "types_mask2_p",
          "EPG+EPGt inhiben": "types_mask6_p", "sin inhibición (no integra)": "types_mask0_p"}
KYMO_AVS = [-0.04, 0.0, 0.04]
CUE_PHASES = list(np.linspace(-np.pi, np.pi, 8, endpoint=False))
NB = 12  # contenedores de fase (50 neuronas de compás: ~4 por contenedor)
T = 200


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def bin_by_angle(r, ang):
    """r: [T, n] -> [T, NB] media por contenedor de fase absoluta (-pi, pi]."""
    idx = np.minimum(((wrap(ang) + np.pi) / (2 * np.pi) * NB).astype(int), NB - 1)
    out = np.zeros((r.shape[0], NB))
    for b in range(NB):
        sel = idx == b
        out[:, b] = r[:, sel].mean(1) if sel.any() else np.nan
    return out


def fwhm_deg(profile):
    """Ancho a media altura (grados) de un perfil circular de NB contenedores centrado en el
    contenedor central; media altura medida sobre el mínimo. Interpolación lineal."""
    p = np.asarray(profile, dtype=float)
    lo, hi = p.min(), p.max()
    if hi - lo < 1e-9:
        return float("nan")
    half = lo + 0.5 * (hi - lo)
    c = int(np.argmax(p))
    step = 360.0 / NB
    width = 0.0
    for direction in (1, -1):
        k = 0
        while k < NB // 2:
            a, b = p[(c + direction * k) % NB], p[(c + direction * (k + 1)) % NB]
            if b < half:
                width += (k + (a - half) / max(a - b, 1e-12)) * step
                break
            k += 1
        else:
            width += (NB // 2) * step
    return float(width)


def main():
    g = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    nodes = g["nodes"]
    tids = torch.as_tensor(pd.factorize(nodes["type"])[0])
    compass = (nodes["type"].isin(["EPG", "EPGt"]) & nodes["ring_angle"].notna()).to_numpy()
    ang = nodes.loc[compass, "ring_angle"].to_numpy()
    edges = (np.arange(NB) + 0.5) / NB * 2 * np.pi - np.pi  # centros de los contenedores
    out = {"n_bins": NB, "bin_centers_rad": edges.tolist(), "kymo_av": KYMO_AVS, "steps": list(range(0, T, 2)),
           "cue_phases_rad": [float(x) for x in CUE_PHASES], "models": {}}
    for name, label in MODELS.items():
        m = CXRingNetwork(g["n_nodes"], g["edge_index"], g["synapse_weight"], tau=10.0, recurrent_gain=2.0,
                          activation="tanh", type_ids=tids, learn_type_params=True)
        m.load_state_dict(torch.load(os.path.join("data", "interim", f"model_{label}.pt"), weights_only=True))
        res = {"label": label, "kymograph": {}, "decoded_phase": {}}
        with torch.no_grad():
            # kymographs
            for av0 in KYMO_AVS:
                av = np.zeros(T)
                av[20:] = av0
                st = m(build_external_input(av, nodes, gain=10.0, cue_phase=0.0, cue_gain=10.0))
                k = bin_by_angle(st.numpy()[:, compass], ang)
                res["kymograph"][f"{av0:+.2f}"] = np.round(k[::2], 3).tolist()
                res["decoded_phase"][f"{av0:+.2f}"] = np.round(decode_heading(st, nodes).numpy()[::2], 3).tolist()
                if av0 == 0.0:
                    r = st.numpy()[:, compass]
                    dec = decode_heading(st, nodes).numpy()
                    prof = np.zeros(NB)
                    n = 0
                    for t in range(40, T):
                        rel = wrap(ang - dec[t])
                        idx = np.minimum(((rel + np.pi) / (2 * np.pi) * NB).astype(int), NB - 1)
                        rt = r[t] - r[t].mean()
                        row = np.array([rt[idx == b].mean() if (idx == b).any() else np.nan for b in range(NB)])
                        if np.isfinite(row).all():
                            prof += row
                            n += 1
                    prof /= max(n, 1)
                    # el pico queda en el contenedor central (fase relativa 0 -> contenedor NB/2)
                    cen = np.roll(prof, NB // 2 - int(np.argmax(prof)))
                    res["profile_rel"] = np.round(prof, 4).tolist()
                    res["fwhm_deg"] = fwhm_deg(cen)
                    res["contrast"] = float((prof.max() - prof.min()) / max(np.abs(r[40:]).mean(), 1e-9))
                    res["frac_saturated"] = float((np.abs(r[40:]) > 0.95).mean())
                    res["rate_min_max"] = [float(prof.min()), float(prof.max())]
            # memoria de fase arbitraria
            err60, err199 = [], []
            for cp in CUE_PHASES:
                st = m(build_external_input(np.zeros(T), nodes, gain=10.0, cue_phase=float(cp), cue_gain=10.0))
                dec = decode_heading(st, nodes).numpy()
                err60.append(float(wrap(dec[60] - cp)))
                err199.append(float(wrap(dec[199] - cp)))
            res["phase_memory"] = {"error_rad_t60": err60, "error_rad_t199": err199,
                                   "mean_abs_error_t60": float(np.mean(np.abs(err60))),
                                   "mean_abs_error_t199": float(np.mean(np.abs(err199))),
                                   "drift_rad_per_100steps": float(np.mean(np.abs(wrap(np.array(err199) - np.array(err60)))) / 139 * 100)}
        out["models"][name] = res
        pm = res["phase_memory"]
        print(f"{name:28s} ancho={res['fwhm_deg']:6.1f} deg  contraste={res['contrast']:.2f}  saturadas={res['frac_saturated']:.2f}  "
              f"error de fase t60={pm['mean_abs_error_t60']:.2f} t199={pm['mean_abs_error_t199']:.2f} rad")
    json.dump(out, open(os.path.join("results", "bump_profile.json"), "w"))


if __name__ == "__main__":
    main()

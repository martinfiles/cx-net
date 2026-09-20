"""
Validación mecanística de las redes que "integran" (entrada (14) del cuaderno): ¿forman un
bump localizado que rota con velocidad proporcional a la entrada, o el ángulo decodificado
es un artefacto del decodificador?
  - localización: |sum(r e^{i th})| / sum|r| sobre las EPG/EPGt (1 = toda la actividad en una
    fase; ~0 = repartida uniformemente), medida tras la pista (t >= 40).
  - ganancia de velocidad: desplazamiento angular decodificado entre t=30 y t=120 frente al
    desplazamiento de rumbo que corresponde a una velocidad constante av (av*90 rad); ganancia
    ideal = 1. Se ajusta una recta por el origen y se da el R^2.
Salida: results/bump_check.json
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
AVS = [-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08]  # dentro del rango de entrenamiento (std 0.15/paso)


def main():
    g = load_cx_graph(ring_source="eb_synapses", ring_sign=1.0)
    nodes = g["nodes"]
    tids = torch.as_tensor(pd.factorize(nodes["type"])[0])
    compass = (nodes["type"].isin(["EPG", "EPGt"]) & nodes["ring_angle"].notna()).to_numpy()
    ang = nodes.loc[compass, "ring_angle"].to_numpy()
    T = 200
    out = {}
    for name, label in MODELS.items():
        m = CXRingNetwork(g["n_nodes"], g["edge_index"], g["synapse_weight"], tau=10.0, recurrent_gain=2.0,
                          activation="tanh", type_ids=tids, learn_type_params=True)
        m.load_state_dict(torch.load(os.path.join("data", "interim", f"model_{label}.pt")))
        vel, loc = [], []
        with torch.no_grad():
            for av0 in AVS:
                av = np.zeros(T)
                av[20:] = av0
                st = m(build_external_input(av, nodes, gain=10.0, cue_phase=0.0, cue_gain=10.0))
                r = st.numpy()[:, compass]
                pv = np.abs((r * np.exp(1j * ang)).sum(1)) / np.maximum(np.abs(r).sum(1), 1e-9)
                loc.append(float(pv[40:].mean()))
                d = np.unwrap(decode_heading(st, nodes).numpy())
                vel.append(float(d[120] - d[30]))  # desplazamiento decodificado (rad)
        vel = np.array(vel)
        a = np.array(AVS) * 90.0  # desplazamiento de rumbo esperado (rad)
        slope = float((a * vel).sum() / (a * a).sum())
        icpt = 0.0
        r2 = float(1 - ((vel - slope * a) ** 2).sum() / max(((vel - vel.mean()) ** 2).sum(), 1e-12))
        out[name] = {"label": label, "av": AVS, "decoded_displacement_rad": vel.tolist(), "gain": float(slope),
                     "intercept": float(icpt), "r2": r2, "localization_mean": float(np.mean(loc))}
        print(f"{name:28s} ganancia={slope:5.2f} R2={r2:5.2f} localizacion={np.mean(loc):.2f} | desplaz. decodificado (rad) "
              f"para esperado -7.2/-1.8/+1.8/+7.2: {vel[0]:+.2f} {vel[3]:+.2f} {vel[5]:+.2f} {vel[-1]:+.2f}")
    json.dump(out, open(os.path.join("results", "bump_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

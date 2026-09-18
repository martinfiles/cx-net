"""
Chequeo de cordura: ¿puede la red sobreajustar UN ÚNICO ensayo fijo?

Antes de gastar tiempo ajustando hiperparámetros para que generalice sobre
ensayos aleatorios (train.py), hay que confirmar que la arquitectura tiene
capacidad suficiente y que el grafo de pérdida/gradiente está bien planteado.
Si esto no baja a near-zero, el problema es de diseño (arquitectura, escala,
tarea), no de variabilidad entre ensayos -- y no tiene sentido seguir
ajustando train.py hasta resolverlo aquí primero.
"""

import json

import torch

from .graph_utils import load_cx_graph
from .model import CXRingNetwork
from .task import build_external_input, circular_loss, decode_heading, generate_trial


def run(n_steps: int = 500, T: int = 200, lr: float = 0.05, tau: float = 5.0,
        recurrent_gain: float = 4.0, seed: int = 42, hold_prob: float = 0.0,
        perturb_amp: float = 0.0, sign_reg: float = 0.0) -> dict:
    torch.manual_seed(0)
    graph = load_cx_graph()
    nodes = graph["nodes"]

    model = CXRingNetwork(
        graph["n_nodes"], graph["edge_index"], graph["synapse_weight"],
        tau=tau, recurrent_gain=recurrent_gain,
    )
    optimizer = torch.optim.Adam([model.sign_param], lr=lr)

    av, heading = generate_trial(T=T, hold_prob=hold_prob, perturb_amp=perturb_amp, seed=seed)
    ext_input = build_external_input(av, nodes)

    loss_history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        states = model(ext_input)
        decoded = decode_heading(states, nodes)
        loss = circular_loss(decoded, heading)
        loss.backward()
        if sign_reg > 0:
            (sign_reg * model.sign_confidence_penalty()).backward()
        torch.nn.utils.clip_grad_norm_([model.sign_param], max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())
        if step % 50 == 0 or step == n_steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    with torch.no_grad():
        s = torch.tanh(model.sign_param)

    return {
        "tau": tau,
        "recurrent_gain": recurrent_gain,
        "lr": lr,
        "hold_prob": hold_prob,
        "perturb_amp": perturb_amp,
        "sign_reg": sign_reg,
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "loss_min": min(loss_history),
        "mean_abs_sign": s.abs().mean().item(),
        "frac_polarized_gt_0.9": (s.abs() > 0.9).float().mean().item(),
        "frac_undecided_lt_0.3": (s.abs() < 0.3).float().mean().item(),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))

"""
Fase 2 (piloto en hemibrain): entrena la red de signo libre sobre la tarea de
integración de rumbo y comprueba que el pipeline converge.

IMPORTANTE: esto NO evalúa todavía H1 (signo aprendido vs. neurotransmisor
real) -- el hemibrain no tiene esa anotación (ver docs/lab-notebook.md). Este
script valida que la arquitectura y la tarea funcionan de extremo a extremo
antes de invertir en la migración a MaleCNS/CAVE, que es donde vive el
ground truth real.
"""

import json
import os

import torch

from .graph_utils import load_cx_graph
from .model import CXRingNetwork
from .task import build_external_input, circular_loss, decode_heading, generate_trial

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim")


def train(n_epochs: int = 300, T: int = 200, lr: float = 0.02, seed: int = 0,
          trials_per_step: int = 8) -> dict:
    torch.manual_seed(seed)
    graph = load_cx_graph()
    nodes = graph["nodes"]

    model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"])
    optimizer = torch.optim.Adam([model.sign_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30
    )

    best_loss = float("inf")
    best_state = None
    loss_history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        batch_loss = 0.0
        for i in range(trials_per_step):
            av, heading = generate_trial(T=T, seed=epoch * trials_per_step + i)
            ext_input = build_external_input(av, nodes)
            states = model(ext_input)
            decoded = decode_heading(states, nodes)
            loss = circular_loss(decoded, heading)
            (loss / trials_per_step).backward()
            batch_loss += loss.item() / trials_per_step

        torch.nn.utils.clip_grad_norm_([model.sign_param], max_norm=1.0)
        optimizer.step()
        scheduler.step(batch_loss)

        loss_history.append(batch_loss)
        if batch_loss < best_loss:
            best_loss = batch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == n_epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:4d}  loss {batch_loss:.4f}  lr {current_lr:.4g}  best {best_loss:.4f}")

    model.load_state_dict(best_state)

    os.makedirs(INTERIM_DIR, exist_ok=True)
    signs = model.learned_signs().numpy()
    torch.save(model.state_dict(), os.path.join(INTERIM_DIR, "model_pilot.pt"))

    results = {
        "n_epochs": n_epochs,
        "T": T,
        "lr": lr,
        "seed": seed,
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "loss_min": min(loss_history),
        "n_edges": len(signs),
        "frac_excitatory_learned": float((signs > 0).mean()),
    }
    with open(os.path.join(INTERIM_DIR, "pilot_results.json"), "w") as f:
        json.dump({"results": results, "loss_history": loss_history}, f, indent=2)

    return results


if __name__ == "__main__":
    results = train(n_epochs=500, lr=0.05)
    print(json.dumps(results, indent=2))

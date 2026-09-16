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
          trials_per_step: int = 16, ema_alpha: float = 0.05, patience: int = 150,
          tau: float = 5.0, recurrent_gain: float = 4.0,
          adam_betas: tuple[float, float] = (0.9, 0.999)) -> dict:
    """
    ema_alpha / patience: el primer intento uso ReduceLROnPlateau directamente
    sobre la pérdida cruda de cada época, que es muy ruidosa (cada época usa
    ensayos aleatorios distintos) -- el scheduler confundió ruido con
    estancamiento y bajó la tasa de aprendizaje a ~0 hacia la época 700 de
    2500, dejando el resto del entrenamiento sin efecto (ver lab-notebook).
    Ahora el scheduler decide sobre una media móvil exponencial de la
    pérdida, no sobre el valor crudo por época.
    """
    torch.manual_seed(seed)
    graph = load_cx_graph()
    nodes = graph["nodes"]

    model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"],
                           tau=tau, recurrent_gain=recurrent_gain)
    optimizer = torch.optim.Adam([model.sign_param], lr=lr, betas=adam_betas)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )

    best_loss = float("inf")
    best_state = None
    loss_history = []
    ema_loss = None
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

        ema_loss = batch_loss if ema_loss is None else (1 - ema_alpha) * ema_loss + ema_alpha * batch_loss
        scheduler.step(ema_loss)

        loss_history.append(batch_loss)
        if batch_loss < best_loss:
            best_loss = batch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == n_epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:4d}  loss {batch_loss:.4f}  ema {ema_loss:.4f}  lr {current_lr:.4g}  best {best_loss:.4f}", flush=True)

        # Guardado periódico: si el proceso se interrumpe (apagado, corte de
        # luz, timeout), no se pierde todo el entrenamiento -- se puede
        # seguir evaluando/reanudando desde el último checkpoint intermedio.
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            os.makedirs(INTERIM_DIR, exist_ok=True)
            torch.save(best_state, os.path.join(INTERIM_DIR, "model_checkpoint_inprogress.pt"))
            with open(os.path.join(INTERIM_DIR, "training_progress.json"), "w") as f:
                json.dump({
                    "epoch": epoch, "n_epochs": n_epochs, "best_loss": best_loss,
                    "ema_loss": ema_loss, "current_lr": current_lr, "done": epoch == n_epochs - 1,
                }, f, indent=2)

    model.load_state_dict(best_state)

    os.makedirs(INTERIM_DIR, exist_ok=True)
    signs = model.learned_signs().numpy()
    torch.save(model.state_dict(), os.path.join(INTERIM_DIR, "model_pilot.pt"))

    with torch.no_grad():
        soft_sign = torch.tanh(model.sign_param)

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
        "mean_abs_sign": soft_sign.abs().mean().item(),
        "frac_polarized_gt_0.9": (soft_sign.abs() > 0.9).float().mean().item(),
    }
    with open(os.path.join(INTERIM_DIR, "pilot_results.json"), "w") as f:
        json.dump({"results": results, "loss_history": loss_history}, f, indent=2)

    return results


if __name__ == "__main__":
    results = train(n_epochs=2500, lr=0.05, trials_per_step=64, patience=150)
    print(json.dumps(results, indent=2))

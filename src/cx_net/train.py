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

import numpy as np
import pandas as pd
import torch

from .evaluate import attach_ground_truth
from .graph_utils import DATA_DIR, load_cx_graph
from .model import CXRingNetwork
from .task import (
    build_external_input, circular_loss, decode_heading, evaluate_on_trials,
    generate_anchored_trial, generate_held_out_set, generate_trial, integration_diagnostics,
)

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim")

# Ver aviso en task.py sobre HELD_OUT_SEED_BASE: debe quedar muy por debajo de
# ese valor para cualquier (seed, n_epochs, trials_per_step) razonable, así
# los ensayos de entrenamiento y validación nunca coinciden por semilla.
SEED_STRIDE = 1_000_000

# Orden de tipos para las máscaras de signo por tipo (control_type_mask).
TYPE_ORDER = ["Delta7", "EPG", "EPGt", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]


def train(n_epochs: int = 300, T: int = 200, lr: float = 0.02, seed: int = 0,
          trials_per_step: int = 16, ema_alpha: float = 0.05, patience: int = 150,
          tau: float = 5.0, recurrent_gain: float = 4.0,
          adam_betas: tuple[float, float] = (0.9, 0.999),
          hold_prob: float = 0.0, perturb_amp: float = 0.0, sign_reg: float = 0.0,
          run_label: str | None = None, anchor: bool = False, max_av: float = 0.08,
          ring_source: str = "glomerulus", ring_sign: float = 1.0,
          activation: str = "tanh", type_params: bool = False, real_sign_control: bool = False,
          in_gain: float = 3.0, cue_gain: float = 3.0,
          control_shuffle_seed: int | None = None,
          control_swap_fraction: float | None = None,
          control_type_mask: int | None = None) -> dict:
    """
    ema_alpha / patience: el primer intento uso ReduceLROnPlateau directamente
    sobre la pérdida cruda de cada época, que es muy ruidosa (cada época usa
    ensayos aleatorios distintos) -- el scheduler confundió ruido con
    estancamiento y bajó la tasa de aprendizaje a ~0 hacia la época 700 de
    2500, dejando el resto del entrenamiento sin efecto (ver lab-notebook).
    Ahora el scheduler decide sobre una media móvil exponencial de la
    pérdida, no sobre el valor crudo por época.

    `seed`: hasta la entrada (11) del cuaderno, `seed` solo controlaba la
    inicialización de `sign_param` -- la secuencia de ensayos de
    entrenamiento dependía únicamente de (epoch, trials_per_step), igual
    para cualquier `seed`. Eso confundía "efecto del hiperparámetro" con
    "qué ensayos le tocaron a esta corrida en particular" al comparar
    configuraciones con una sola repetición. Ahora `seed` también desplaza
    la secuencia de ensayos (vía SEED_STRIDE), así que correr el mismo config
    con distintos `seed` da variación genuina de inicialización Y de datos,
    útil para medir varianza entre corridas.

    `sign_reg` (2026-09-18): en vez de seguir buscando una variante de
    TAREA que fuerce polarización de forma emergente (agotado tras
    `hold_prob`/`perturb_amp`, ver lab-notebook), esto añade una presión
    directa sobre el PARÁMETRO -- `model.sign_confidence_penalty()` -- que
    no depende de los ensayos ni de la dinámica de la red. Se suma al
    gradiente de tarea en cada paso, así que compite con él: si `sign_reg`
    es demasiado alto, puede fijar el signo de una arista en su dirección
    de inicialización (ruido) antes de que el gradiente de tarea tenga
    ocasión de corregirla -- por eso se sigue el mismo protocolo de
    dosis-respuesta que con `perturb_amp` antes de aceptar cualquier valor.

    `anchor` / `max_av` / `ring_source` / `ring_sign` (2026-09-19, entradas
    (4)-(6) del cuaderno; `max_av` solo se usa con `anchor=True`): tarea anclada (fase inicial aleatoria + pista breve) y
    `ring_angle` medido en el EB. Con los valores por defecto se reproduce
    exactamente el comportamiento anterior.

    `activation` / `type_params` / `in_gain` / `cue_gain` (entrada (8)):
    activación rectificada, parámetros por tipo celular (ver `model.py`) y
    ganancias de la entrada de velocidad y de la pista. `real_sign_control=True`
    es SOLO un control de realizabilidad: fija los signos al neurotransmisor
    real y entrena únicamente los parámetros por tipo; nunca debe usarse para
    obtener nada que se le pase a un aprendiz de signos.
    """
    torch.manual_seed(seed)
    graph = load_cx_graph(ring_source=ring_source, ring_sign=ring_sign)
    nodes = graph["nodes"]

    type_ids = torch.as_tensor(pd.factorize(nodes["type"])[0]) if type_params else None
    type_names = list(pd.factorize(nodes["type"])[1]) if type_params else None
    model = CXRingNetwork(graph["n_nodes"], graph["edge_index"], graph["synapse_weight"],
                           tau=tau, recurrent_gain=recurrent_gain, activation=activation,
                           type_ids=type_ids, learn_type_params=type_params)
    if real_sign_control:
        if not type_params:
            raise ValueError("real_sign_control sin type_params no entrena nada")
        gt = attach_ground_truth(nodes, DATA_DIR)
        true_node_sign = gt["expected_sign"].to_numpy()
        node_sign = true_node_sign.copy()
        if control_type_mask is not None:
            # patrón de signo POR TIPO: bit i de la máscara = 1 -> los tipos TYPE_ORDER[i] inhiben
            # (máscara 1 = solo Delta7 inhibe = la asignación real de neurotransmisor)
            node_sign = np.array([-1.0 if (control_type_mask >> TYPE_ORDER.index(t)) & 1 else 1.0
                                  for t in nodes["type"]])
        elif control_swap_fraction is not None:
            # dosis-respuesta: se elige una fracción de neuronas y se permutan sus etiquetas
            # entre sí (mismo recuento 110/42); solo cambian las que reciben la otra etiqueta
            rng = np.random.default_rng(control_shuffle_seed or 0)
            idx = rng.choice(len(node_sign), size=int(round(control_swap_fraction * len(node_sign))), replace=False)
            node_sign[idx] = rng.permutation(node_sign[idx])
        elif control_shuffle_seed is not None:  # control: mismo recuento 110/42, reparto barajado entre neuronas
            node_sign = np.random.default_rng(control_shuffle_seed).permutation(node_sign)
        src_idx = graph["edge_index"][0].numpy()
        real_edge_sign = node_sign[src_idx]
        frac_nodes_changed = float((node_sign != true_node_sign).mean())
        frac_edges_changed = float((real_edge_sign != true_node_sign[src_idx]).mean())
        with torch.no_grad():
            model.sign_param.copy_(torch.as_tensor(real_edge_sign, dtype=torch.float32) * 10.0)
        model.sign_param.requires_grad_(False)
        sign_reg = 0.0
    trainable = ([] if real_sign_control else [model.sign_param]) + model.type_parameters()
    optimizer = torch.optim.Adam(trainable, lr=lr, betas=adam_betas)
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
            trial_seed = seed * SEED_STRIDE + epoch * trials_per_step + i
            if anchor:
                av, heading, theta0 = generate_anchored_trial(
                    T=T, max_av=max_av, hold_prob=hold_prob, perturb_amp=perturb_amp, seed=trial_seed)
            else:
                av, heading = generate_trial(T=T, hold_prob=hold_prob, perturb_amp=perturb_amp, seed=trial_seed)
                theta0 = None
            ext_input = build_external_input(av, nodes, gain=in_gain, cue_phase=theta0, cue_gain=cue_gain)
            states = model(ext_input)
            decoded = decode_heading(states, nodes)
            loss = circular_loss(decoded, heading)
            (loss / trials_per_step).backward()
            batch_loss += loss.item() / trials_per_step

        reg_loss = 0.0
        if sign_reg > 0:
            reg = sign_reg * model.sign_confidence_penalty()
            reg.backward()
            reg_loss = reg.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
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

    # Pérdida sobre el set de validación FIJO (30 ensayos, semillas reservadas
    # -- ver task.py) para poder comparar configuraciones/semillas de forma
    # limpia, sin que la comparación esté sesgada por qué ensayos de
    # ENTRENAMIENTO le tocaron a cada corrida (ver entrada 11 del cuaderno).
    held_out_trials = generate_held_out_set(T=T, hold_prob=hold_prob, perturb_amp=perturb_amp,
                                            anchor=anchor, max_av=max_av)
    held_out_loss = evaluate_on_trials(model, nodes, held_out_trials, in_gain=in_gain, cue_gain=cue_gain)
    diagnostics = integration_diagnostics(model, nodes, held_out_trials, in_gain=in_gain,
                                          cue_gain=cue_gain) if anchor else None

    os.makedirs(INTERIM_DIR, exist_ok=True)
    signs = model.learned_signs().numpy()
    label = run_label or "pilot"
    torch.save(model.state_dict(), os.path.join(INTERIM_DIR, f"model_{label}.pt"))

    with torch.no_grad():
        soft_sign = torch.tanh(model.sign_param)

    results = {
        "run_label": label,
        "n_epochs": n_epochs,
        "T": T,
        "lr": lr,
        "seed": seed,
        "trials_per_step": trials_per_step,
        "hold_prob": hold_prob,
        "perturb_amp": perturb_amp,
        "sign_reg": sign_reg,
        "anchor": anchor,
        "max_av": max_av,
        "activation": activation,
        "type_params": type_params,
        "real_sign_control": real_sign_control,
        "control_shuffle_seed": control_shuffle_seed,
        "control_swap_fraction": control_swap_fraction,
        "control_type_mask": control_type_mask,
        "frac_nodes_changed": frac_nodes_changed if real_sign_control else None,
        "frac_edges_changed": frac_edges_changed if real_sign_control else None,
        "in_gain": in_gain,
        "cue_gain": cue_gain,
        "diagnostics": diagnostics,
        "type_names": type_names,
        "log_pair_gain": model.log_pair_gain.detach().tolist() if type_params else None,
        "type_bias": model.type_bias.detach().tolist() if type_params else None,
        "log_input_scale": float(model.log_input_scale) if type_params else None,
        "ring_source": ring_source,
        "ring_sign": ring_sign,
        "tau": tau,
        "recurrent_gain": recurrent_gain,
        "adam_betas": list(adam_betas),
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "loss_min": min(loss_history),
        "held_out_loss": held_out_loss,
        "n_edges": len(signs),
        "frac_excitatory_learned": float((signs > 0).mean()),
        "mean_abs_sign": soft_sign.abs().mean().item(),
        "frac_polarized_gt_0.9": (soft_sign.abs() > 0.9).float().mean().item(),
    }
    with open(os.path.join(INTERIM_DIR, f"results_{label}.json"), "w") as f:
        json.dump({"results": results, "loss_history": loss_history}, f, indent=2)

    return results


if __name__ == "__main__":
    results = train(n_epochs=2500, lr=0.05, trials_per_step=64, patience=150)
    print(json.dumps(results, indent=2))

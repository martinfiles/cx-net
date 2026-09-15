"""
Tarea de integración de rumbo: la red recibe una traza de velocidad angular
inyectada como entrada externa asimétrica sobre las neuronas PEN_a/PEN_b
(izquierda +, derecha -, o viceversa -- una simplificación estándar en
modelos de este circuito: en la mosca real esa asimetría llega desde otras
regiones que no forman parte de este subcircuito acotado) y debe mantener,
en la población "brújula" (EPG/EPGt), un patrón de actividad cuya posición
(leída como vector poblacional sobre `ring_angle`) seguya el rumbo real
obtenido al integrar esa velocidad angular en el tiempo.
"""

import numpy as np
import torch


def generate_trial(T: int = 200, max_av: float = 0.08, seed: int | None = None):
    """Devuelve (velocidad_angular[T], rumbo_real[T]) en radianes."""
    rng = np.random.default_rng(seed)
    av = rng.normal(0.0, max_av, size=T)
    heading = np.cumsum(av)
    return av, heading


def build_external_input(av: np.ndarray, nodes, gain: float = 3.0) -> torch.Tensor:
    """[T, n_nodes]: la velocidad angular solo entra por PEN_a/PEN_b, con signo
    opuesto entre hemisferios (L empuja el bump en un sentido, R en el otro)."""
    T = len(av)
    n_nodes = len(nodes)
    ext = torch.zeros(T, n_nodes)
    drive_mask_L = ((nodes["type"].isin(["PEN_a(PEN1)", "PEN_b(PEN2)"])) & (nodes["hemisphere"] == "L")).to_numpy()
    drive_mask_R = ((nodes["type"].isin(["PEN_a(PEN1)", "PEN_b(PEN2)"])) & (nodes["hemisphere"] == "R")).to_numpy()
    av_t = torch.tensor(av, dtype=torch.float32).unsqueeze(1)  # [T, 1]
    ext[:, drive_mask_L] = gain * av_t
    ext[:, drive_mask_R] = -gain * av_t
    return ext


def decode_heading(states: torch.Tensor, nodes) -> torch.Tensor:
    """Vector poblacional sobre la población brújula (EPG/EPGt) en cada paso."""
    compass_mask = nodes["type"].isin(["EPG", "EPGt"]).to_numpy() & nodes["ring_angle"].notna().to_numpy()
    angles = torch.tensor(nodes.loc[compass_mask, "ring_angle"].to_numpy(), dtype=torch.float32)
    r_compass = states[:, compass_mask]  # [T, n_compass]
    x = (r_compass * torch.cos(angles)).sum(dim=1)
    y = (r_compass * torch.sin(angles)).sum(dim=1)
    return torch.atan2(y, x)  # [T]


def circular_loss(decoded: torch.Tensor, target: torch.Tensor, warmup: int = 20) -> torch.Tensor:
    """1 - cos(diff), promediado tras un breve calentamiento para que el bump se forme."""
    target_wrapped = torch.atan2(torch.sin(torch.as_tensor(target, dtype=torch.float32)),
                                  torch.cos(torch.as_tensor(target, dtype=torch.float32)))
    diff = decoded[warmup:] - target_wrapped[warmup:]
    return (1 - torch.cos(diff)).mean()

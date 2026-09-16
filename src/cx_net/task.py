"""
Tarea de integración de rumbo: la red recibe una traza de velocidad angular
inyectada como entrada externa asimétrica sobre las neuronas PEN_a/PEN_b
(izquierda +, derecha -, o viceversa -- una simplificación estándar en
modelos de este circuito: en la mosca real esa asimetría llega desde otras
regiones que no forman parte de este subcircuito acotado) y debe mantener,
en la población "brújula" (EPG/EPGt), un patrón de actividad cuya posición
(leída como vector poblacional sobre `ring_angle`) seguya el rumbo real
obtenido al integrar esa velocidad angular en el tiempo.

`hold_prob` (2026-09-17, ver lab-notebook entrada 2026-09-17): con
`hold_prob=0`, la tarea original de Fase 2 permite que muchas asignaciones
de signo distintas logren un desempeño agregado similar -- el barrido
multi-semilla mostró polarización débil de forma consistente sin importar
el hiperparámetro de entrenamiento, probablemente porque la tarea no
restringe lo suficiente el signo de cada arista. Con `hold_prob > 0`, la
traza de velocidad angular incluye tramos de "quietud" (av=0) intercalados
-- durante esos tramos la red debe MANTENER el bump de rumbo sin ninguna
entrada externa que la ayude, apoyándose solo en su propia dinámica
recurrente (persistent activity de un ring attractor), lo que depende mucho
más directamente de una estructura de signo correcta (en particular, de
inhibición lateral tipo Delta7 para evitar que el bump se disperse) que la
integración pura. Mismo `circular_loss` de siempre -- no se inventa un
término de pérdida nuevo, solo se hace la tarea más exigente.
"""

import numpy as np
import torch


def generate_trial(T: int = 200, max_av: float = 0.08, hold_prob: float = 0.0,
                    hold_block: int = 20, seed: int | None = None):
    """Devuelve (velocidad_angular[T], rumbo_real[T]) en radianes.

    Con `hold_prob > 0`: se divide T en bloques de `hold_block` pasos: cada
    bloque tiene probabilidad `hold_prob` de ser un tramo de quietud (av=0
    forzado), simulando a la mosca parada -- el rumbo debe mantenerse sin
    entrada externa. `hold_prob=0` (default) reproduce exactamente el
    comportamiento original (compatibilidad con Fase 2)."""
    rng = np.random.default_rng(seed)
    av = rng.normal(0.0, max_av, size=T)
    if hold_prob > 0:
        n_blocks = T // hold_block
        for b in range(n_blocks):
            if rng.random() < hold_prob:
                av[b * hold_block:(b + 1) * hold_block] = 0.0
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


# Rango de semillas reservado para el set de validación held-out: disjunto de
# cualquier semilla que pueda generarse durante entrenamiento (train.py usa
# seed_run * SEED_STRIDE + epoch * trials_per_step + i, con SEED_STRIDE muy
# por debajo de este rango), para que ningún ensayo de validación pueda
# colarse como ensayo de entrenamiento por coincidencia de semilla.
HELD_OUT_SEED_BASE = 900_000_000


def generate_held_out_set(n_trials: int = 30, T: int = 200, hold_prob: float = 0.0):
    """Conjunto FIJO de ensayos de validación: mismas semillas siempre, para
    poder comparar configuraciones/semillas de entrenamiento entre sí sin que
    la comparación esté confundida por qué ensayos le tocaron a cada una."""
    return [generate_trial(T=T, hold_prob=hold_prob, seed=HELD_OUT_SEED_BASE + i) for i in range(n_trials)]


def evaluate_on_trials(model, nodes, trials) -> float:
    """Pérdida media (sin gradiente) sobre un conjunto de ensayos fijo."""
    with torch.no_grad():
        total = 0.0
        for av, heading in trials:
            ext_input = build_external_input(av, nodes)
            states = model(ext_input)
            decoded = decode_heading(states, nodes)
            total += circular_loss(decoded, heading).item()
        return total / len(trials)

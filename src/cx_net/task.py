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

`perturb_amp` (2026-09-17, segunda condición simultánea a `hold_prob`,
ver lab-notebook): `hold_prob` sube `mean_abs_sign` de 0.307 a 0.364 pero
no alcanza el umbral -- la entrada del cuaderno del mismo día concluye que
hace falta un eje de exigencia distinto, no otra variante del mismo
(`hold_prob=0.5` no mejora sobre 0.3). `perturb_amp` añade ruido de alta
frecuencia y media cero al canal de entrada externa (PEN_a/PEN_b) que NO
forma parte del rumbo objetivo (el `heading` para la pérdida se integra
solo a partir de la velocidad angular "verdadera", limpia). La red debe
seguir integrando la señal real Y, simultáneamente, rechazar el ruido para
que el bump no se desvíe ni se disperse -- eso depende de que la dinámica
recurrente (filtrado temporal vía `tau` + inhibición lateral correcta) esté
bien puesta, distinto del mecanismo que ejercita `hold_prob` (memoria
persistente sin entrada). El ruido se aplica en TODOS los pasos, incluidos
los tramos de quietud de `hold_prob` -- combinar ambos exige sostener el
bump sin ayuda Y filtrando ruido al mismo tiempo, la condición más
exigente probada hasta ahora.

Tarea anclada (2026-09-19, ver lab-notebook entrada (6)): la tarea original
pide el rumbo ABSOLUTO empezando en 0, sin ninguna pista de dónde está el 0.
Con heading0=0 y velocidades pequeñas, un decodificador constante en 0 daba
pérdida 0.18 (3-4 veces mejor que los modelos entrenados) y un integrador
perfecto con fase inicial arbitraria daba ~1.0: la tarea no medía integración.
`generate_anchored_trial` sortea una fase inicial theta0 ~ U(-pi, pi) por
ensayo; `build_external_input(..., cue_phase=theta0)` la inyecta como pulso
breve sobre las neuronas de compás (EPG/EPGt) en los primeros `cue_steps`
pasos, y el objetivo pasa a ser theta0 + integral de la velocidad. Ahora el
decodificador constante da ~1.0 y "recordar theta0 sin integrar" da una
pérdida que depende de `max_av` (0.55 con 0.15); la red tiene que anclar,
mantener e integrar.
"""

import numpy as np
import torch


def generate_trial(T: int = 200, max_av: float = 0.08, hold_prob: float = 0.0,
                    hold_block: int = 20, perturb_amp: float = 0.0,
                    seed: int | None = None):
    """Devuelve (velocidad_angular_de_manejo[T], rumbo_real[T]) en radianes.

    Con `hold_prob > 0`: se divide T en bloques de `hold_block` pasos: cada
    bloque tiene probabilidad `hold_prob` de ser un tramo de quietud (av=0
    forzado), simulando a la mosca parada -- el rumbo debe mantenerse sin
    entrada externa. `hold_prob=0` (default) reproduce exactamente el
    comportamiento original (compatibilidad con Fase 2).

    Con `perturb_amp > 0`: se añade ruido gaussiano iid (media 0, desvío
    `perturb_amp`) a la velocidad angular DESPUÉS de calcular `heading` --
    el ruido llega a la red (vía `build_external_input`) pero NO cuenta para
    el rumbo objetivo, así que la red debe integrarlo sin dejarse arrastrar
    por él. `perturb_amp=0` (default) reproduce el comportamiento original.
    """
    rng = np.random.default_rng(seed)
    av_true = rng.normal(0.0, max_av, size=T)
    if hold_prob > 0:
        n_blocks = T // hold_block
        for b in range(n_blocks):
            if rng.random() < hold_prob:
                av_true[b * hold_block:(b + 1) * hold_block] = 0.0
    heading = np.cumsum(av_true)
    av_drive = av_true
    if perturb_amp > 0:
        av_drive = av_true + rng.normal(0.0, perturb_amp, size=T)
    return av_drive, heading


ANCHOR_SEED_OFFSET = 500_000_000


def generate_anchored_trial(T: int = 200, max_av: float = 0.15, hold_prob: float = 0.0,
                             hold_block: int = 20, perturb_amp: float = 0.0,
                             seed: int = 0):
    """(velocidad_angular, rumbo_real, theta0). Igual que `generate_trial` pero
    el rumbo parte de una fase inicial aleatoria theta0 (sorteada con una
    semilla derivada, para no alterar la secuencia de `generate_trial`)."""
    av, heading = generate_trial(T=T, max_av=max_av, hold_prob=hold_prob,
                                  hold_block=hold_block, perturb_amp=perturb_amp, seed=seed)
    theta0 = float(np.random.default_rng(ANCHOR_SEED_OFFSET + seed).uniform(-np.pi, np.pi))
    return av, heading + theta0, theta0


def build_external_input(av: np.ndarray, nodes, gain: float = 3.0,
                         cue_phase: float | None = None, cue_steps: int = 20,
                         cue_gain: float = 3.0) -> torch.Tensor:
    """[T, n_nodes]: la velocidad angular solo entra por PEN_a/PEN_b, con signo
    opuesto entre hemisferios (L empuja el bump en un sentido, R en el otro).

    Con `cue_phase` (tarea anclada): durante los primeros `cue_steps` pasos se
    suma a las neuronas de compás (EPG/EPGt con `ring_angle`) la entrada
    cue_gain * max(0, cos(ring_angle - cue_phase)), un pulso con forma de bump
    centrado en la fase inicial."""
    T = len(av)
    n_nodes = len(nodes)
    ext = torch.zeros(T, n_nodes)
    if cue_phase is not None:
        compass = nodes["type"].isin(["EPG", "EPGt"]).to_numpy() & nodes["ring_angle"].notna().to_numpy()
        ang = torch.tensor(nodes.loc[compass, "ring_angle"].to_numpy(), dtype=torch.float32)
        ext[:cue_steps, compass] = cue_gain * torch.clamp(torch.cos(ang - cue_phase), min=0.0)
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
    # atan2(0, 0) tiene gradiente nan en algunas versiones de torch (2.4.1) y 0 en
    # otras (2.14). Se sustituye el vector nulo por uno constante (sin gradiente):
    # valor idéntico (atan2(0, 0) = 0) y gradiente 0 en cualquier versión; los
    # vectores no nulos no cambian ni un bit.
    zero = (x == 0) & (y == 0)
    x = torch.where(zero, torch.ones_like(x), x)
    y = torch.where(zero, torch.zeros_like(y), y)
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


def generate_held_out_set(n_trials: int = 30, T: int = 200, hold_prob: float = 0.0,
                           perturb_amp: float = 0.0, anchor: bool = False, max_av: float = 0.08):
    """Conjunto FIJO de ensayos de validación: mismas semillas siempre, para
    poder comparar configuraciones/semillas de entrenamiento entre sí sin que
    la comparación esté confundida por qué ensayos le tocaron a cada una."""
    if anchor:
        return [generate_anchored_trial(T=T, max_av=max_av, hold_prob=hold_prob, perturb_amp=perturb_amp,
                                        seed=HELD_OUT_SEED_BASE + i) for i in range(n_trials)]
    return [generate_trial(T=T, hold_prob=hold_prob, perturb_amp=perturb_amp, seed=HELD_OUT_SEED_BASE + i)
            for i in range(n_trials)]


def evaluate_on_trials(model, nodes, trials, in_gain: float = 3.0, cue_gain: float = 3.0) -> float:
    """Pérdida media (sin gradiente) sobre un conjunto de ensayos fijo."""
    with torch.no_grad():
        total = 0.0
        for trial in trials:
            av, heading = trial[0], trial[1]
            theta0 = trial[2] if len(trial) > 2 else None  # tarea anclada: (av, rumbo, theta0)
            ext_input = build_external_input(av, nodes, gain=in_gain, cue_phase=theta0, cue_gain=cue_gain)
            states = model(ext_input)
            decoded = decode_heading(states, nodes)
            total += circular_loss(decoded, heading).item()
        return total / len(trials)


def integration_diagnostics(model, nodes, trials, in_gain: float = 3.0, cue_gain: float = 3.0) -> dict:
    """Solo tarea anclada: pérdida, error de anclaje a t=20 (grados) y pendiente
    sum(dd*hc)/sum(hc*hc) entre el cambio decodificado dd y el cambio real hc
    desde t=20 (1 = integra a velocidad correcta, 0 = no integra)."""
    import math
    losses, anchor_err, num, den = [], [], 0.0, 0.0
    with torch.no_grad():
        for av, heading, theta0 in trials:
            states = model(build_external_input(av, nodes, gain=in_gain, cue_phase=theta0, cue_gain=cue_gain))
            dec = decode_heading(states, nodes)
            losses.append(circular_loss(dec, heading).item())
            d = np.unwrap(dec.numpy())
            anchor_err.append(abs(math.degrees(np.angle(np.exp(1j * (d[20] - theta0))))))
            dd = d[20:] - d[20]
            hc = heading[20:] - heading[20]
            num += float((dd * hc).sum())
            den += float((hc * hc).sum())
    return {"loss": float(np.mean(losses)), "anchor_err_deg": float(np.mean(anchor_err)),
            "slope": num / max(den, 1e-9)}

"""
Figuras del informe (PNG, 1200x675, tema claro). Lee SOLO de `results/` (versionado),
así que se pueden regenerar sin `data/interim/`. Salida: docs/figures/fig{1..5}.png.

Convenciones (guía de visualización del proyecto): un color por función;
azul = serie principal, naranja = "asignación real de neurotransmisor" en TODAS las
figuras, gris = neutro; marcas finas; etiquetas directas selectivas; texto siempre en
tintas de texto (nunca en el color de la serie). Paleta (huecos 1-3) validada con
validate_palette.js: todos los pares pasan; el aguamarina queda por debajo de 3:1 de
contraste, por eso se usa solo con etiquetas visibles.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b2"
MEMORY = 0.466  # held-out de "recordar theta0 sin integrar"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 11})


def frame(title, subtitle, source="cx-net · datos: male-cns:v1.0 (neuPrint)"):
    fig = plt.figure(figsize=(8, 4.5), dpi=150)
    fig.text(0.05, 0.94, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.875, subtitle, fontsize=10.5, color=INK2, va="top")
    fig.text(0.05, 0.03, source, fontsize=8, color=MUTED)
    return fig


def clean(ax, grid_axis="x"):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def dot(ax, x, y, color, r=7):
    ax.plot(x, y, "o", color=color, markersize=r, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, clip_on=False)


# ---------------------------------------------------------------- fig 1
def fig1():
    runs = pd.read_csv(os.path.join(RES, "runs_summary.csv"))
    tr = runs[runs.label.str.match(r"^signreg05_seed\d$")].held_out_loss.to_numpy()
    fig = frame("Un decodificador inerte supera a todos los modelos entrenados",
                "Pérdida en el conjunto held-out de la tarea original (menor es mejor)")
    ax = fig.add_axes([0.30, 0.16, 0.65, 0.62])
    clean(ax)
    rows = [("Decodificador constante\n(siempre apunta a 0)", [0.179], INK2),
            ("Modelos entrenados\n(8 semillas)", tr, BLUE),
            ("Fase aleatoria\n(la «referencia de azar»)", [0.98], GRAY)]
    for i, (lab, xs, c) in enumerate(rows):
        y = 2 - i
        for j, x in enumerate(sorted(xs)):
            dot(ax, x, y + (0.13 if j % 2 else -0.13) * (len(xs) > 1), c)
        ax.text(-0.02, y, lab, ha="right", va="center", fontsize=10.5, color=INK, transform=ax.get_yaxis_transform())
    ax.text(0.179 + 0.03, 2, "0.18", va="center", fontsize=10.5, color=INK)
    ax.text(tr.max() + 0.03, 1, f"{tr.min():.2f}–{tr.max():.2f}", va="center", fontsize=10.5, color=INK)
    ax.text(0.98 + 0.03, 0, "≈1.0", va="center", fontsize=10.5, color=INK)
    ax.set_xlim(-0.02, 1.15)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("pérdida held-out (1 − cos del error de rumbo)", fontsize=9.5)
    return fig


# ---------------------------------------------------------------- fig 2
def fig2():
    g = pd.read_csv(os.path.join(RES, "epg_glomerulus_angles.csv"))
    fig = frame("Los dos hemisferios recorren el anillo en sentidos opuestos",
                "Fase angular de cada glomérulo (EPG); el radio crece de L1/R1 (dentro) a L8/R8 (fuera)")
    ax = fig.add_axes([0.08, 0.11, 0.50, 0.63], projection="polar")
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 9.2)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=9, color=MUTED)
    ax.grid(color=GRID, linewidth=1)
    ax.spines["polar"].set_color(GRID)
    for h, c in (("L", BLUE), ("R", ORANGE)):
        s = g[g.hemisphere == h].sort_values("glomerulus")
        ang = np.unwrap(np.radians(s.mean_angle_deg.to_numpy()))
        rad = s.glomerulus.to_numpy().astype(float)
        ax.plot(ang, rad, "-", color=c, linewidth=2, solid_capstyle="round", zorder=2)
        for a, r in zip(ang, rad):
            ax.plot(a, r, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(ang[-1], rad[-1] + 0.95, f"{h}8", color=INK, fontsize=9, ha="center", va="center")
    # leyenda (2 series) con la marca de color al lado del texto
    for i, (lab, c) in enumerate((("Hemisferio izquierdo (L)", BLUE), ("Hemisferio derecho (R)", ORANGE))):
        y = 0.62 - i * 0.09
        fig.patches.append(plt.Rectangle((0.63, y - 0.012), 0.025, 0.024, transform=fig.transFigure, color=c))
        fig.text(0.67, y, lab, fontsize=10.5, color=INK, va="center")
    fig.text(0.63, 0.38, "Cada hemisferio cubre el anillo\ncompleto en pasos de ≈45°.\n\nLos glomérulos homólogos L y R\nquedan desfasados ≈22°, y L y R\ngiran en sentido contrario\n(un espejo).",
             fontsize=10, color=INK2, va="top", linespacing=1.4)
    return fig


# ---------------------------------------------------------------- fig 3
def fig3():
    t = json.load(open(os.path.join(RES, "types_analysis.json")))
    t = sorted(t, key=lambda x: x["held_out"])
    ho = np.array([x["held_out"] for x in t])
    integra = np.array([x["integra"] for x in t])
    real_rank = [i for i, x in enumerate(t) if x["mask"] == 1][0] + 1
    n_int = int(integra.sum())
    fig = frame(f"{n_int} de 64 patrones integran; la asignación real es uno más",
                "Pérdida held-out de cada patrón (una corrida por patrón), ordenados de mejor a peor")
    ax = fig.add_axes([0.09, 0.16, 0.86, 0.62])
    clean(ax, "y")
    ax.axhline(MEMORY, color=INK2, linewidth=1, zorder=1)
    ax.text(0.8, MEMORY + 0.03, "memoria sin integrar (0.466)", ha="left", fontsize=9.5, color=INK2)
    xs = np.arange(1, 65)
    ax.scatter(xs[integra], ho[integra], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="integra")
    ax.scatter(xs[~integra], ho[~integra], s=46, color=GRAY, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="no integra")
    ax.scatter([real_rank], [ho[real_rank - 1]], s=120, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4,
               label="asignación real")
    ax.annotate(f"asignación real: puesto {real_rank} de 64 (p = {real_rank}/64 = {real_rank/64:.2f})",
                (real_rank, ho[real_rank - 1]), xytext=(real_rank + 4, 0.03), fontsize=10, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1, shrinkA=2, shrinkB=6))
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 0.78)
    ax.set_xticks([1, 16, 32, 48, 64])
    ax.set_xlabel("patrón (ordenado por pérdida)", fontsize=9.5)
    ax.set_ylabel("pérdida held-out", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=10, handletextpad=0.3, labelcolor=INK, bbox_to_anchor=(0.0, 1.02))
    return fig


# ---------------------------------------------------------------- fig 4
def fig4():
    t = json.load(open(os.path.join(RES, "types_analysis.json")))
    names = ["Delta7", "EPG", "EPGt", "PEG", "PEN_a", "PEN_b"]
    exc, inh = [], []
    for i in range(6):
        e = [x["integra"] for x in t if not (x["mask"] >> i) & 1]
        h = [x["integra"] for x in t if (x["mask"] >> i) & 1]
        exc.append(100 * np.mean(e))
        inh.append(100 * np.mean(h))
    order = np.argsort(np.array(inh) - np.array(exc))  # menor diferencia abajo
    fig = frame("Ningún signo por tipo basta; pesan más PEN_b y Delta7",
                "Porcentaje de patrones que integran según el signo asignado a cada tipo (32 patrones por punto)")
    ax = fig.add_axes([0.16, 0.19, 0.78, 0.59])
    clean(ax)
    for k, i in enumerate(order):
        ax.plot([exc[i], inh[i]], [k, k], color=GRID, linewidth=3, zorder=1, solid_capstyle="round")
        dot(ax, exc[i], k, BLUE)
        dot(ax, inh[i], k, AQUA)
    ax.set_yticks(range(6))
    ax.set_yticklabels([names[i] for i in order], fontsize=10.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 5.6)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    top = list(order).index(0)
    ax.text(exc[0] - 2, top, f"{exc[0]:.0f} %", ha="right", va="center", fontsize=10, color=INK)
    ax.text(inh[0] + 2, top, f"{inh[0]:.0f} %", ha="left", va="center", fontsize=10, color=INK)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=8, label="tipo excitador"),
                       Line2D([], [], marker="o", linestyle="", color=AQUA, markersize=8, label="tipo inhibidor")],
              loc="lower right", frameon=False, fontsize=10, labelcolor=INK, handletextpad=0.3)
    kb = list(order).index(5)
    ax.text(exc[5] + 2, kb, f"{exc[5]:.0f} %", ha="left", va="center", fontsize=10, color=INK)
    ax.text(inh[5] - 2, kb, f"{inh[5]:.0f} %", ha="right", va="center", fontsize=10, color=INK)
    return fig


# ---------------------------------------------------------------- fig 5
def fig5():
    r = json.load(open(os.path.join(RES, "rates_by_type.json")))
    bins = np.array(r["bins"])
    mid = (bins[:-1] + bins[1:]) / 2
    fig = frame("La solución con la química real usa Delta7 con tasa negativa",
                "Distribución de la tasa de activación de las neuronas Delta7 (activación tanh, t ≥ 40)")
    ax = fig.add_axes([0.09, 0.19, 0.86, 0.59])
    clean(ax, "y")
    for key, lab, c in (("sin_inhibicion", "sin inhibición (no integra)", BLUE), ("real", "asignación real (integra)", ORANGE)):
        d = r["models"][key]["types"]["Delta7"]
        h = np.array(d["hist"], dtype=float)
        h = 100 * h / h.sum()
        ax.stairs(h, bins, color=c, linewidth=2, fill=False, zorder=3)
        ax.stairs(h, bins, color=c, alpha=0.10, fill=True, zorder=2)
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    real = r["models"]["real"]["types"]["Delta7"]["frac_positive"]
    ax.set_xlim(-1, 1)
    ax.set_xlabel("tasa de activación (negativa = no fisiológica)", fontsize=9.5)
    ax.set_ylabel("% de instantes", fontsize=9.5)
    ax.text(-0.97, ax.get_ylim()[1] * 0.93, f"asignación real: {100*(1-real):.0f} % del tiempo\ncon tasa negativa",
            fontsize=10.5, color=INK, va="top")
    for x0, lab, c in ((0.55, "asignación real (integra)", ORANGE), (0.55, "sin inhibición (no integra)", BLUE)):
        pass
    for i, (lab, c) in enumerate((("asignación real (integra)", ORANGE), ("sin inhibición (no integra)", BLUE))):
        y = 0.72 - i * 0.075
        fig.patches.append(plt.Rectangle((0.62, y - 0.011), 0.022, 0.022, transform=fig.transFigure, color=c))
        fig.text(0.65, y, lab, fontsize=10, color=INK, va="center")
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for i, f in enumerate((fig1, fig2, fig3, fig4, fig5), 1):
        fig = f()
        fig.savefig(os.path.join(OUT, f"fig{i}.png"))
        plt.close(fig)
        print(f"fig{i}.png")


if __name__ == "__main__":
    main()

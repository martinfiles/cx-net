"""
Figuras del informe (PNG, 1200x675, tema claro), en español e inglés. Lee SOLO de `results/`
(versionado), así que se pueden regenerar sin `data/interim/`.
Salida: docs/figures/fig{1..5}.png (es) y docs/figures/en/fig{1..5}.png (en).
Uso: python -m src.cx_net.make_figures [--lang es|en|all]

Convenciones (guía de visualización del proyecto): un color por función;
azul = serie principal, naranja = "asignación real de neurotransmisor" en TODAS las
figuras, gris = neutro; marcas finas; etiquetas directas selectivas; texto siempre en
tintas de texto (nunca en el color de la serie). Paleta (huecos 1-3) validada con
validate_palette.js: todos los pares pasan; el aguamarina queda por debajo de 3:1 de
contraste, por eso se usa solo con etiquetas visibles.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(ROOT, "results")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b2"
MEMORY = 0.466  # held-out de "recordar theta0 sin integrar"

TEXT = {
    "es": {
        "source": "cx-net · datos: male-cns:v1.0 (neuPrint)",
        "pct": "{v:.0f} %", "pct_axis": ["0 %", "25 %", "50 %", "75 %", "100 %"],
        "f1_title": "Un decodificador inerte supera a todos los modelos entrenados",
        "f1_sub": "Pérdida held-out de la tarea original, sin pista de fase (menor es mejor)",
        "f1_const": "Decodificador constante\n(siempre apunta a 0)", "f1_trained": "Modelos entrenados\n(8 semillas)",
        "f1_random": "Fase aleatoria\n(la «referencia de azar»)", "f1_x": "pérdida held-out (1 − cos del error de rumbo)",
        "f2_title": "Los dos hemisferios recorren el anillo en sentidos opuestos",
        "f2_sub": "Fase angular de cada glomérulo (EPG); el radio crece de L1/R1 (dentro) a L8/R8 (fuera)",
        "f2_L": "Hemisferio izquierdo (L)", "f2_R": "Hemisferio derecho (R)",
        "f2_note": "Cada hemisferio cubre el anillo\ncompleto en pasos de ≈45°.\n\nLos glomérulos homólogos L y R\nquedan desfasados ≈22°, y L y R\ngiran en sentido contrario\n(un espejo).",
        "f3_title": "{n} de 64 patrones integran; la asignación real es uno más",
        "f3_sub": "Pérdida held-out de cada patrón (una corrida por patrón), ordenados de mejor a peor",
        "f3_mem": "memoria sin integrar (0.466)", "f3_real": "asignación real: puesto {r} de 64 (p = {r}/64 = {p:.2f})",
        "f3_yes": "integra", "f3_no": "no integra", "f3_leg_real": "asignación real",
        "f3_x": "patrón (ordenado por pérdida)", "f3_y": "pérdida held-out",
        "f4_title": "Ningún signo por tipo basta; PEN_b y Delta7 se asocian más",
        "f4_sub": "Porcentaje de patrones que integran según el signo asignado a cada tipo (32 patrones por punto)",
        "f4_exc": "tipo excitador", "f4_inh": "tipo inhibidor",
        "f5_title": "La solución con la química real usa Delta7 con tasa negativa",
        "f5_sub": "Distribución de la tasa de activación de las neuronas Delta7 (activación tanh, t ≥ 40)",
        "f5_x": "tasa de activación (negativa = por debajo de la basal)", "f5_y": "% de instantes",
        "f5_ann": "asignación real: {v:.0f} % del tiempo\ncon tasa negativa",
        "f5_real": "asignación real (integra)", "f5_none": "sin inhibición (no integra)",
    },
    "en": {
        "source": "cx-net · data: male-cns:v1.0 (neuPrint)",
        "pct": "{v:.0f}%", "pct_axis": ["0%", "25%", "50%", "75%", "100%"],
        "f1_title": "An inert decoder beats every trained model",
        "f1_sub": "Held-out loss on the original task, no phase cue (lower is better)",
        "f1_const": "Constant decoder\n(always points to 0)", "f1_trained": "Trained models\n(8 seeds)",
        "f1_random": "Random phase\n(the “chance” baseline)", "f1_x": "held-out loss (1 − cos of heading error)",
        "f2_title": "The two hemispheres traverse the ring in opposite directions",
        "f2_sub": "Angular phase of each glomerulus (EPG); radius grows from L1/R1 (inside) to L8/R8 (outside)",
        "f2_L": "Left hemisphere (L)", "f2_R": "Right hemisphere (R)",
        "f2_note": "Each hemisphere covers the full\nring in ≈45° steps.\n\nHomologous L and R glomeruli are\noffset by ≈22°, and L and R turn\nin opposite directions\n(a mirror image).",
        "f3_title": "{n} of 64 patterns integrate; the real assignment is just one",
        "f3_sub": "Held-out loss of each pattern (one run per pattern), sorted best to worst",
        "f3_mem": "memory without integrating (0.466)", "f3_real": "real assignment: rank {r} of 64 (p = {r}/64 = {p:.2f})",
        "f3_yes": "integrates", "f3_no": "does not integrate", "f3_leg_real": "real assignment",
        "f3_x": "pattern (sorted by loss)", "f3_y": "held-out loss",
        "f4_title": "No per-type sign is enough; PEN_b and Delta7 stand out",
        "f4_sub": "Percentage of patterns that integrate, by the sign assigned to each type (32 patterns per point)",
        "f4_exc": "excitatory type", "f4_inh": "inhibitory type",
        "f5_title": "The real-chemistry solution runs Delta7 at negative rates",
        "f5_sub": "Distribution of Delta7 activation rate (tanh activation, t ≥ 40)",
        "f5_x": "activation rate (negative = below baseline)", "f5_y": "% of time steps",
        "f5_ann": "real assignment: {v:.0f}% of the time\nat a negative rate",
        "f5_real": "real assignment (integrates)", "f5_none": "no inhibition (does not integrate)",
    },
}
L = TEXT["es"]  # idioma activo (se fija en main)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 11})


def frame(title, subtitle):
    fig = plt.figure(figsize=(8, 4.5), dpi=150)
    fig.text(0.05, 0.94, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.875, subtitle, fontsize=10.5, color=INK2, va="top")
    fig.text(0.05, 0.03, L["source"], fontsize=8, color=MUTED)
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
    fig = frame(L["f1_title"], L["f1_sub"])
    ax = fig.add_axes([0.30, 0.16, 0.65, 0.62])
    clean(ax)
    rows = [(L["f1_const"], [0.179], INK2), (L["f1_trained"], tr, BLUE), (L["f1_random"], [0.98], GRAY)]
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
    ax.set_xlabel(L["f1_x"], fontsize=9.5)
    return fig


# ---------------------------------------------------------------- fig 2
def fig2():
    g = pd.read_csv(os.path.join(RES, "epg_glomerulus_angles.csv"))
    fig = frame(L["f2_title"], L["f2_sub"])
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
    for i, (lab, c) in enumerate(((L["f2_L"], BLUE), (L["f2_R"], ORANGE))):
        y = 0.62 - i * 0.09
        fig.patches.append(plt.Rectangle((0.63, y - 0.012), 0.025, 0.024, transform=fig.transFigure, color=c))
        fig.text(0.67, y, lab, fontsize=10.5, color=INK, va="center")
    fig.text(0.63, 0.38, L["f2_note"], fontsize=10, color=INK2, va="top", linespacing=1.4)
    return fig


# ---------------------------------------------------------------- fig 3
def fig3():
    t = json.load(open(os.path.join(RES, "types_analysis.json")))
    t = sorted(t, key=lambda x: x["held_out"])
    ho = np.array([x["held_out"] for x in t])
    integra = np.array([x["integra"] for x in t])
    real_rank = [i for i, x in enumerate(t) if x["mask"] == 1][0] + 1
    n_int = int(integra.sum())
    fig = frame(L["f3_title"].format(n=n_int), L["f3_sub"])
    ax = fig.add_axes([0.09, 0.16, 0.86, 0.62])
    clean(ax, "y")
    ax.axhline(MEMORY, color=INK2, linewidth=1, zorder=1)
    ax.text(0.8, MEMORY + 0.03, L["f3_mem"], ha="left", fontsize=9.5, color=INK2)
    xs = np.arange(1, 65)
    ax.scatter(xs[integra], ho[integra], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=L["f3_yes"])
    ax.scatter(xs[~integra], ho[~integra], s=46, color=GRAY, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=L["f3_no"])
    ax.scatter([real_rank], [ho[real_rank - 1]], s=120, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4,
               label=L["f3_leg_real"])
    ax.annotate(L["f3_real"].format(r=real_rank, p=real_rank / 64), (real_rank, ho[real_rank - 1]),
                xytext=(real_rank + 4, 0.03), fontsize=10, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1, shrinkA=2, shrinkB=6))
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 0.78)
    ax.set_xticks([1, 16, 32, 48, 64])
    ax.set_xlabel(L["f3_x"], fontsize=9.5)
    ax.set_ylabel(L["f3_y"], fontsize=9.5)
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
    fig = frame(L["f4_title"], L["f4_sub"])
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
    ax.set_xticklabels(L["pct_axis"])
    top = list(order).index(0)
    ax.text(exc[0] - 2, top, L["pct"].format(v=exc[0]), ha="right", va="center", fontsize=10, color=INK)
    ax.text(inh[0] + 2, top, L["pct"].format(v=inh[0]), ha="left", va="center", fontsize=10, color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=8, label=L["f4_exc"]),
                       Line2D([], [], marker="o", linestyle="", color=AQUA, markersize=8, label=L["f4_inh"])],
              loc="lower right", frameon=False, fontsize=10, labelcolor=INK, handletextpad=0.3)
    kb = list(order).index(5)
    ax.text(exc[5] + 2, kb, L["pct"].format(v=exc[5]), ha="left", va="center", fontsize=10, color=INK)
    ax.text(inh[5] - 2, kb, L["pct"].format(v=inh[5]), ha="right", va="center", fontsize=10, color=INK)
    return fig


# ---------------------------------------------------------------- fig 5
def fig5():
    r = json.load(open(os.path.join(RES, "rates_by_type.json")))
    bins = np.array(r["bins"])
    fig = frame(L["f5_title"], L["f5_sub"])
    ax = fig.add_axes([0.09, 0.19, 0.86, 0.59])
    clean(ax, "y")
    for key, c in (("sin_inhibicion", BLUE), ("real", ORANGE)):
        d = r["models"][key]["types"]["Delta7"]
        h = np.array(d["hist"], dtype=float)
        h = 100 * h / h.sum()
        ax.stairs(h, bins, color=c, linewidth=2, fill=False, zorder=3)
        ax.stairs(h, bins, color=c, alpha=0.10, fill=True, zorder=2)
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    real = r["models"]["real"]["types"]["Delta7"]["frac_positive"]
    ax.set_xlim(-1, 1)
    ax.set_xlabel(L["f5_x"], fontsize=9.5)
    ax.set_ylabel(L["f5_y"], fontsize=9.5)
    ax.text(-0.97, ax.get_ylim()[1] * 0.93, L["f5_ann"].format(v=100 * (1 - real)), fontsize=10.5, color=INK, va="top")
    for i, (lab, c) in enumerate(((L["f5_real"], ORANGE), (L["f5_none"], BLUE))):
        y = 0.72 - i * 0.075
        fig.patches.append(plt.Rectangle((0.60, y - 0.011), 0.022, 0.022, transform=fig.transFigure, color=c))
        fig.text(0.63, y, lab, fontsize=10, color=INK, va="center")
    return fig


def main():
    global L
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", choices=["es", "en", "all"])
    args = ap.parse_args()
    for lang in (["es", "en"] if args.lang == "all" else [args.lang]):
        L = TEXT[lang]
        out = os.path.join(ROOT, "docs", "figures", "" if lang == "es" else "en")
        os.makedirs(out, exist_ok=True)
        for i, f in enumerate((fig1, fig2, fig3, fig4, fig5), 1):
            fig = f()
            fig.savefig(os.path.join(out, f"fig{i}.png"))
            plt.close(fig)
        print(f"{lang}: 5 figuras en {os.path.normpath(out)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CREADS Fig. 2 — Training–deployment workflow (paper format).

Two horizontal tracks (TRAINING / DEPLOYMENT), rounded white boxes with bold
titles + grey detail lines, a dashed group "estimated without the outcome",
vertical dashed hand-off arrows, and two italic footnotes.
All numbers follow CREADS_Revised_Manuscript_4.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"]:
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
avail = {f.name for f in font_manager.fontManager.ttflist}
BASE = "Liberation Sans" if "Liberation Sans" in avail else "DejaVu Sans"
plt.rcParams.update({"font.family": BASE, "axes.linewidth": 0.6})

import os
from pathlib import Path
ROOT = os.environ.get("CREADS_ROOT", str(Path(__file__).resolve().parent))
os.makedirs(ROOT, exist_ok=True)

# ----------------------------------------------------------------- canvas
fig = plt.figure(figsize=(13.2, 4.6), dpi=300)          # panoramic ~2.7:1
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

INK = "#1a1a1a"; GREY = "#4a4a4a"; FAINT = "#666666"
BOX_W, BOX_H = 13.0, 24.0
CX = [13.0 + i * 15.7 for i in range(6)]                 # 6 aligned columns
Y_TR, Y_DE = 71.0, 27.0
ASPECT_FIX = (13.2 / 100) / (4.9 / 100)                  # x-unit / y-unit in inches


def box(cx, cy, title, *details):
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.55,rounding_size=0.55",
        mutation_aspect=ASPECT_FIX, fc="white", ec=INK, lw=1.0, zorder=3))
    n = len(details)
    block = (1 + n) * 4.4 + 1.2
    top = cy + block / 2
    ax.text(cx, top, title, ha="center", va="top", fontsize=8.0,
            fontweight="bold", color=INK, zorder=4)
    for j, d in enumerate(details):
        ax.text(cx, top - 5.2 * (j + 1), d, ha="center", va="top",
                fontsize=6.0, color=GREY, zorder=4)


def harrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1,
                                mutation_scale=13), zorder=2)


def varrow(x, label):
    ax.annotate("", xy=(x, Y_DE + BOX_H / 2 + 0.4),
                xytext=(x, Y_TR - BOX_H / 2 - 0.4),
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.0,
                                linestyle=(0, (3, 3)), mutation_scale=11),
                zorder=2)
    ax.text(x + 0.75, (Y_TR + Y_DE) / 2 - 0.5, label, ha="left", va="center",
            fontsize=6.4, style="italic", color=GREY, zorder=4)


# ------------------------------------------------------- rotated row labels
ax.text(2.6, Y_TR, "TRAINING", rotation=90, ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=INK)
ax.text(2.6, Y_DE, "DEPLOYMENT", rotation=90, ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=INK)

# ============================================================ TRAINING row
tr = [
    ("Labelled regions",
     "340 regions \u00b7 21 countries", "HDI observed, 2023"),
    ("Harmonisation",
     "polygon alignment", "standardised preprocessing"),
    ("Latent structure",
     "accessibility GMM \u00b7 K = 3", "4 country clusters \u00b7 10 regime cells"),
    ("Functional representation",
     "multi-scale profiles", "regime-specific FPCA"),
    ("Shared base learner",
     "RF \u00b7 GB \u00b7 Ridge ensemble", "country-demeaned target"),
    ("Two-pass corrections",
     "shrinkage-weighted", "regime-specific"),
]
for cx, (t, *d) in zip(CX, tr):
    box(cx, Y_TR, t, *d)
for i in range(5):
    harrow(CX[i] + BOX_W / 2 + 0.8, CX[i + 1] - BOX_W / 2 - 0.8, Y_TR)

# dashed group: outcome-independent blocks (columns 2-4)
gx0, gx1 = CX[1] - BOX_W / 2 - 1.5, CX[3] + BOX_W / 2 + 1.5
gy0, gy1 = Y_TR - BOX_H / 2 - 1.5, Y_TR + BOX_H / 2 + 1.5
ax.add_patch(FancyBboxPatch(
    (gx0, gy0), gx1 - gx0, gy1 - gy0,
    boxstyle="round,pad=0.4,rounding_size=0.5", mutation_aspect=ASPECT_FIX,
    fc="none", ec=FAINT, lw=1.0, linestyle=(0, (4, 3)), zorder=1))
ax.text((gx0 + gx1) / 2, gy1 + 1.0, "estimated without the outcome",
        ha="center", va="bottom", fontsize=6.8, style="italic", color=GREY)

# ========================================================== DEPLOYMENT row
de = [
    ("Inventory regions",
     "584 regions \u00b7 26 countries", "no HDI label required"),
    ("Same feature schema",
     "identical preprocessing"),
    ("Posterior projection",
     "country cluster + regime type"),
    ("Same functional basis",
     "learned in training"),
    ("Same two-pass estimator",
     "frozen weights & corrections"),
    ("HDI prediction & targeting",
     "point estimate + interval"),
]
for cx, (t, *d) in zip(CX, de):
    box(cx, Y_DE, t, *d)
for i in range(5):
    harrow(CX[i] + BOX_W / 2 + 0.8, CX[i + 1] - BOX_W / 2 - 0.8, Y_DE)

# ------------------------------------------------- vertical hand-off arrows
varrow(CX[1], "same schema")
varrow(CX[2], "same structure")
varrow(CX[3], "same basis")
varrow(CX[4], "frozen weights")

fig.savefig(f"{ROOT}/fig2_CREADS_workflow.png", dpi=300, facecolor="white")
print("SAVED fig2_CREADS_workflow.png")
from PIL import Image
Image.open(f"{ROOT}/fig2_CREADS_workflow.png").convert("RGB").save(
    f"{ROOT}/fig2_CREADS_workflow.pdf", "PDF", resolution=300.0, quality=95)
print("SAVED fig2_CREADS_workflow.pdf")

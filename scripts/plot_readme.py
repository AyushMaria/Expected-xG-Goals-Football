"""Banner and pitch map for the README (docs/assets/).

    python scripts/plot_readme.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Rectangle

GRASS, GRASS2, LINE = "#0e3b2c", "#114634", "#e9f2ec"
GOLD = "#f2c14e"
XG_CMAP = LinearSegmentedColormap.from_list("xg", ["#1f5f8b", "#3fa7a0", "#f2c14e", "#eb6834", "#e23b3b"])


def attacking_half(ax, stripes=True):
    """Attacking half of a 120x80 StatsBomb pitch, goal on the right (x = 120)."""
    ax.set_facecolor(GRASS)
    if stripes:
        for i, x0 in enumerate(np.arange(60, 120, 6)):
            if i % 2:
                ax.add_patch(Rectangle((x0, 0), 6, 80, color=GRASS2, zorder=0, linewidth=0))
    kw = dict(color=LINE, linewidth=1.6, zorder=2, alpha=0.85)
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], **kw)
    ax.plot([102, 102, 120], [18, 62, 62], **kw)
    ax.plot([102, 120], [18, 18], **kw)
    ax.plot([114, 114, 120], [30, 50, 50], **kw)
    ax.plot([114, 120], [30, 30], **kw)
    ax.plot([120, 121.5, 121.5, 120], [36, 36, 44, 44], **kw)
    ax.scatter([108], [40], s=8, color=LINE, zorder=2)
    ax.add_patch(Arc((108, 40), 20, 20, theta1=127, theta2=233, fill=False, **kw))
    ax.add_patch(Arc((60, 40), 20, 20, theta1=-90, theta2=90, fill=False, **kw))
    ax.set_xlim(58, 123)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")


def final_third(ax):
    """Final third, goal at the top: horizontal axis = pitch y (0-80), vertical = pitch x (84-120)."""
    ax.set_facecolor(GRASS)
    for i, v0 in enumerate(np.arange(84, 120, 6)):
        if i % 2:
            ax.add_patch(Rectangle((0, v0), 80, 6, color=GRASS2, zorder=0, linewidth=0))
    kw = dict(color=LINE, linewidth=1.6, zorder=2, alpha=0.85)
    ax.plot([0, 0, 80, 80], [84, 120, 120, 84], **kw)
    ax.plot([18, 18, 62, 62], [120, 102, 102, 120], **kw)
    ax.plot([30, 30, 50, 50], [120, 114, 114, 120], **kw)
    ax.plot([36, 36, 44, 44], [120, 121.8, 121.8, 120], color=LINE, linewidth=2.4, zorder=2)
    ax.scatter([40], [108], s=8, color=LINE, zorder=2)
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, fill=False, **kw))
    ax.set_xlim(-1, 81)
    ax.set_ylim(83.5, 122.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    out = Path("docs/assets")
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet("results/oof_xg.parquet")

    # ---------------- banner: every open-play goal, coloured by its xG
    g = oof[oof.goal == 1].sample(frac=1, random_state=1)
    fig = plt.figure(figsize=(16, 5.2), facecolor=GRASS)
    ax = fig.add_axes([0.40, 0.04, 0.58, 0.92])
    final_third(ax)
    g = g[g["x location shot"] >= 84]
    ax.scatter(g["y location shot"], g["x location shot"], c=g.xg, cmap=XG_CMAP, vmin=0, vmax=0.8,
               s=9, alpha=0.8, linewidths=0, zorder=3)
    t = fig.add_axes([0, 0, 0.38, 1])
    t.set_facecolor(GRASS)
    t.axis("off")
    t.text(0.1, 0.70, "Expected Goals", color="white", fontsize=40, fontweight="bold", va="center")
    t.text(0.1, 0.55, "How good was that chance, really?", color=GOLD, fontsize=17, va="center", style="italic")
    t.text(0.1, 0.36, f"{len(oof):,} open-play shots  ·  {oof.game_id.nunique():,} matches", color=LINE, fontsize=13)
    t.text(0.1, 0.28, "StatsBomb open data  ·  men's football since 2000", color=LINE, fontsize=13, alpha=0.8)
    t.text(0.1, 0.12, f"Right: {len(g):,} open-play goals, coloured by their xG", color=LINE, fontsize=10, alpha=0.6)
    # small colour key
    k = fig.add_axes([0.1 * 0.38 + 0.0, 0.05, 0.12, 0.025])
    k.imshow(np.linspace(0, 1, 256)[None, :], cmap=XG_CMAP, aspect="auto")
    k.set_xticks([0, 255], ["xG 0", "0.8+"])
    k.tick_params(colors=LINE, labelsize=8, length=0)
    k.set_yticks([])
    for s in k.spines.values():
        s.set_visible(False)
    fig.savefig(out / "banner.png", dpi=110, facecolor=GRASS)
    plt.close(fig)

    # ---------------- xG map: average xG of shots from each part of the pitch
    fig, ax = plt.subplots(figsize=(9, 7.2), facecolor=GRASS)
    d = oof[oof["x location shot"] >= 80]
    attacking_half(ax, stripes=False)
    hb = ax.hexbin(d["x location shot"], d["y location shot"], C=d.xg, reduce_C_function=np.mean,
                   gridsize=(26, 26), extent=(80, 120, 0, 80), mincnt=8, cmap=XG_CMAP, vmin=0, vmax=0.6,
                   linewidths=0.2, edgecolors=GRASS, zorder=1)
    ax.set_xlim(78, 123)
    ax.set_ylim(6, 74)
    cb = fig.colorbar(hb, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.set_label("Average xG of a shot from here", color=LINE)
    cb.ax.tick_params(colors=LINE)
    cb.outline.set_visible(False)
    ax.set_title("Where the good chances are", color="white", fontsize=15, fontweight="bold", loc="left")
    fig.savefig(out / "xg_map.png", dpi=110, facecolor=GRASS, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

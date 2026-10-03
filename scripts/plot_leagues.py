"""Chart for docs/LEAGUES.md: log-loss change vs the pooled model, per league, with 95% CIs.

    python scripts/plot_leagues.py --results results --out docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference palette, validated with scripts/validate_palette.js (aqua needs labels: markers + direct labels used)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1d1d1b", "#5f5e5a", "#e6e5df"
SERIES = [("own", "Own league only", "#eb6834", "s"),
          ("lolo", "Other three leagues only", "#2a78d6", "o"),
          ("league", "Pooled + league as a feature", "#1baf7a", "D")]
LEAGUES = ["Premier League", "La Liga", "Serie A", "Ligue 1"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--model", default="CatBoost")
    args = ap.parse_args(argv)
    r = json.loads((Path(args.results) / "leagues.json").read_text())[args.model]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    offsets = [-0.22, 0, 0.22]
    for (key, label, color, marker), off in zip(SERIES, offsets):
        for i, lg in enumerate(LEAGUES):
            v = r[lg]["variants"][key]
            d = v["log_loss_minus_pooled"] * 1000
            lo, hi = (x * 1000 for x in v["log_loss_minus_pooled_ci"])
            y = i + off
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(d, y, marker=marker, markersize=8, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle="none", zorder=3, label=label if i == 0 else None)
        # direct label on the last league row
        v = r[LEAGUES[-1]]["variants"][key]
        ax.annotate(label, (v["log_loss_minus_pooled_ci"][1] * 1000, len(LEAGUES) - 1 + off),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0, -0.75, "same as pooled  ", ha="right", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(LEAGUES)), LEAGUES)
    ax.set_ylim(len(LEAGUES) - 0.45, -0.95)
    ax.set_xlabel("Change in log loss vs. the pooled model (×1000) — right of the line is worse", color=INK_2, fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Splitting by league makes xG worse; league info adds nothing",
                 loc="left", color=INK, fontsize=13, fontweight="bold", pad=26)
    ax.text(0, 1.03, f"{args.model}, 2015/16 seasons, held-out matches, 95% intervals from resampling matches",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=9, labelcolor=INK, handletextpad=0.4, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(out / "leagues.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {out}/leagues.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibration charts for docs/RESULTS.md.

    python scripts/plot_results.py --shots data/shots_all_men.parquet --results results --out docs/figures

1. calibration.png   - the best model vs StatsBomb xG on the held-out matches
2. barcelona_only.png - the thesis setup (trained on Barcelona matches only) vs the pooled model,
                        on held-out shots from matches without Barcelona
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xg.model import FEATURES, GROUP, TARGET, calibration_table, modelling_set, split_by_match  # noqa: E402

# reference palette (validated: scripts/validate_palette.js "#2a78d6,#eb6834" --mode light)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1d1d1b", "#5f5e5a", "#e6e5df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BINS = 15


def _style(ax, title, subtitle):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Predicted xG (mean per group of shots)", color=INK_2, fontsize=10)
    ax.set_ylabel("Actual goal rate", color=INK_2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=10)


def _curve(ax, y, p, color, label, lim, dy=0):
    t = calibration_table(y, p, bins=BINS)
    ax.plot(t["mean_xg"], t["goal_rate"], color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    last = t.iloc[-1]
    ax.annotate(label, (last["mean_xg"], last["goal_rate"]), xytext=(6, dy), textcoords="offset points",
                color=INK, fontsize=10, va="center")


def _diagonal(ax, lim):
    ax.plot([0, lim], [0, lim], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(lim * 0.97, lim * 0.97, "perfect calibration", color=INK_2, fontsize=9, ha="right", va="bottom", rotation=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shots", default="data/shots_all_men.parquet")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="docs/figures")
    args = p.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = json.loads((Path(args.results) / "results.json").read_text())
    best = res["best_model"]
    model = joblib.load(Path(args.results) / f"model_{best.replace(' ', '_').lower()}.joblib")

    d = modelling_set(pd.read_parquet(args.shots))
    train, test = split_by_match(d, seed=42)
    p_best = model.predict_proba(test[FEATURES])[:, 1]

    # 1. best model vs StatsBomb on all held-out shots
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    lim = 0.6
    _diagonal(ax, lim)
    _curve(ax, test[TARGET], p_best, BLUE, f"{best} (this project)", lim, dy=-9)
    _curve(ax, test[TARGET], test["official xg"], ORANGE, "StatsBomb xG", lim, dy=9)
    _style(ax, "Predicted xG matches the actual goal rate",
           f"{len(test):,} open-play shots in {test[GROUP].nunique()} held-out matches, {BINS} equal-size groups")
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    # 2. thesis setup (Barcelona-only training) vs pooled, on matches without Barcelona
    barca_games = set(d.loc[d["Team Name"].eq("Barcelona"), GROUP])
    tr_b = train[train[GROUP].isin(barca_games)]
    m_b = clone(model).fit(tr_b[FEATURES], tr_b[TARGET])
    other = test[~test[GROUP].isin(barca_games)]
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    _diagonal(ax, lim)
    _curve(ax, other[TARGET], model.predict_proba(other[FEATURES])[:, 1], BLUE, "Trained on all men's matches", lim, dy=9)
    _curve(ax, other[TARGET], m_b.predict_proba(other[FEATURES])[:, 1], ORANGE, "Trained on Barcelona matches only", lim, dy=-9)
    _style(ax, "A Barcelona-only model overrates other teams' chances",
           f"{len(other):,} held-out open-play shots from matches without Barcelona")
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out / "barcelona_only.png", dpi=150, facecolor=SURFACE); plt.close(fig)
    print(f"wrote {out}/calibration.png and {out}/barcelona_only.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

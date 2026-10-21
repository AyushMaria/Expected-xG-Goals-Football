"""Export the trained models to a compact JSON file for the browser demo (ayushmaria.github.io/xg).

The JSON holds the preprocessing (one-hot categories, scaler), the logistic regression
coefficients, the CatBoost trees, and test vectors so the JavaScript port can be checked.

    python scripts/export_web_model.py --out ../ayushmaria.github.io/xg/model.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xg.model import CATEGORICAL, FEATURES, NUMERIC, modelling_set, split_by_match  # noqa: E402


def r(x, n=6):
    return float(f"{x:.{n}g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--shots", default="data/shots_all_men.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=300)
    a = ap.parse_args()
    res = Path(a.results)

    lr = joblib.load(res / "model_logistic_regression.joblib")
    cb = joblib.load(res / "model_catboost.joblib")
    pre = lr.named_steps["pre"]
    oh, sc = pre.named_transformers_["cat"], pre.named_transformers_["num"]

    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "cb.json"
        cb.named_steps["clf"].save_model(str(p), format="json")
        cbj = json.loads(p.read_text())
    trees = []
    for tr in cbj["oblivious_trees"]:
        trees.append([[s["float_feature_index"] for s in tr["splits"]],
                      [r(s["border"], 8) for s in tr["splits"]],
                      [r(v, 6) for v in tr["leaf_values"]]])
    scale, bias = cbj["scale_and_bias"]

    d = modelling_set(pd.read_parquet(a.shots))
    _, te = split_by_match(d, seed=42)
    sample = te.sample(a.n_test, random_state=0)
    tests = []
    p_lr = lr.predict_proba(sample[FEATURES])[:, 1]
    p_cb = cb.predict_proba(sample[FEATURES])[:, 1]
    for (_, row), a1, a2 in zip(sample.iterrows(), p_lr, p_cb):
        tests.append({"x": {c: (row[c] if c in CATEGORICAL else float(row[c])) for c in FEATURES},
                      "lr": float(a1), "cb": float(a2)})

    out = {
        "source": "github.com/AyushMaria/Expected-xG-Goals-Football (results/model_*.joblib)",
        "categorical": {c: [str(v) for v in cats] for c, cats in zip(CATEGORICAL, oh.categories_)},
        "numeric": NUMERIC,
        "scaler": {"mean": [r(v, 8) for v in sc.mean_], "scale": [r(v, 8) for v in sc.scale_]},
        "lr": {"coef": [r(v, 8) for v in lr.named_steps["clf"].coef_[0]],
               "intercept": r(lr.named_steps["clf"].intercept_[0], 8)},
        "cb": {"scale": scale, "bias": bias[0] if isinstance(bias, list) else bias, "trees": trees},
        "stats": {"shots": int(len(d)), "goals": int(d.goal.sum()), "matches": int(d.game_id.nunique()),
                  "goal_rate": r(d.goal.mean(), 4)},
        "tests": tests,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out}: {Path(a.out).stat().st_size/1e3:.0f} kB, {len(trees)} trees")


if __name__ == "__main__":
    main()

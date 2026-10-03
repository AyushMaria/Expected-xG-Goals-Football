"""Does league style bias a pooled xG model? (step 4)

    python scripts/league_comparison.py --shots data/shots_all_men.parquet --results results

Uses the four complete 2015/16 seasons (Premier League, La Liga, Serie A, Ligue 1). For each
league, every variant is scored on the same held-out matches of that league (5 match folds):

  own      - trained on that league only
  pooled   - trained on all four leagues
  lolo     - trained on the other three leagues only (leave one league out)
  league   - pooled, with the league as an extra feature

Also scores a model trained on all four leagues on the Indian Super League 2021/22.
95% confidence intervals resample matches. Writes results/leagues.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xg.compare import log_loss_diff_ci, per_shot_log_loss, ratio_ci  # noqa: E402
from xg.model import CATEGORICAL, FEATURES, GROUP, NUMERIC, TARGET, modelling_set  # noqa: E402

LEAGUES = ["Premier League", "La Liga", "Serie A", "Ligue 1"]
VARIANTS = ["own", "pooled", "lolo", "league"]


def make_model(kind: str, params: dict, with_league: bool) -> Pipeline:
    cats = CATEGORICAL + (["competition"] if with_league else [])
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                             ("num", StandardScaler(), NUMERIC)])
    if kind == "CatBoost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(verbose=0, random_seed=42, thread_count=2, **params)
    else:
        clf = LogisticRegression(max_iter=5000, **params)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_predict(kind, params, train, test, with_league=False):
    cols = FEATURES + (["competition"] if with_league else [])
    m = make_model(kind, params, with_league).fit(train[cols], train[TARGET])
    return m.predict_proba(test[cols])[:, 1]


def run(kind: str, params: dict, d4: pd.DataFrame, isl: pd.DataFrame) -> dict:
    out = {}
    for league in LEAGUES:
        t0 = time.time()
        L, others = d4[d4["competition"].eq(league)], d4[~d4["competition"].eq(league)]
        preds = {v: np.zeros(len(L)) for v in VARIANTS}
        preds["lolo"] = fit_predict(kind, params, others, L)
        for tr_i, te_i in GroupKFold(n_splits=5).split(L, groups=L[GROUP]):
            tr, te = L.iloc[tr_i], L.iloc[te_i]
            pooled_train = pd.concat([others, tr])
            preds["own"][te_i] = fit_predict(kind, params, tr, te)
            preds["pooled"][te_i] = fit_predict(kind, params, pooled_train, te)
            preds["league"][te_i] = fit_predict(kind, params, pooled_train, te, with_league=True)
        y, g = L[TARGET].to_numpy(), L[GROUP].to_numpy()
        res = {"shots": int(len(L)), "goals": int(y.sum()), "matches": int(L[GROUP].nunique()), "variants": {}}
        for v, p in preds.items():
            r, lo, hi = ratio_ci(g, y, p)
            res["variants"][v] = {"log_loss": float(per_shot_log_loss(y, p).mean()), "auc": float(roc_auc_score(y, p)),
                                  "xg_over_goals": r, "xg_over_goals_ci": [lo, hi]}
            if v != "pooled":   # paired difference vs the pooled model (negative = this variant is better)
                dd, dlo, dhi = log_loss_diff_ci(g, y, p, preds["pooled"])
                res["variants"][v]["log_loss_minus_pooled"] = dd
                res["variants"][v]["log_loss_minus_pooled_ci"] = [dlo, dhi]
        sb, slo, shi = ratio_ci(g, y, L["official xg"].to_numpy())
        res["statsbomb_xg_over_goals"], res["statsbomb_xg_over_goals_ci"] = sb, [slo, shi]
        out[league] = res
        print(f"  {kind} {league}: {time.time() - t0:.0f}s", flush=True)

    # transfer to a very different league and level
    p = fit_predict(kind, params, d4, isl)
    y, g = isl[TARGET].to_numpy(), isl[GROUP].to_numpy()
    r, lo, hi = ratio_ci(g, y, p)
    sb, slo, shi = ratio_ci(g, y, isl["official xg"].to_numpy())
    out["Indian Super League 2021/22 (trained on the four leagues)"] = {
        "shots": int(len(isl)), "goals": int(y.sum()), "matches": int(isl[GROUP].nunique()),
        "log_loss": float(per_shot_log_loss(y, p).mean()), "auc": float(roc_auc_score(y, p)),
        "xg_over_goals": r, "xg_over_goals_ci": [lo, hi],
        "statsbomb_xg_over_goals": sb, "statsbomb_xg_over_goals_ci": [slo, shi]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", default="data/shots_all_men.parquet")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    tuned = json.loads((Path(args.results) / "results.json").read_text())["models"]
    d = modelling_set(pd.read_parquet(args.shots))
    d4 = d[d["season_id"].eq("2015/2016") & d["competition"].isin(LEAGUES)].reset_index(drop=True)
    isl = d[d["competition"].eq("Indian Super league")].reset_index(drop=True)
    print(f"four leagues: {len(d4):,} shots in {d4[GROUP].nunique()} matches; ISL: {len(isl):,} shots", flush=True)
    res = {"data": {"shots": int(len(d4)), "matches": int(d4[GROUP].nunique()), "isl_shots": int(len(isl))}}
    for kind in ["Logistic Regression", "CatBoost"]:
        res[kind] = run(kind, tuned[kind]["best_params"], d4, isl)
    (Path(args.results) / "leagues.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {args.results}/leagues.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

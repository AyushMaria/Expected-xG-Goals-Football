"""Team and player tables from out-of-fold xG (card #6), written to docs/TABLES.md.

    python scripts/xg_tables.py --oof results/oof_xg.parquet --out docs/TABLES.md

Needs results/oof_xg.parquet from scripts/train_evaluate.py and the cached StatsBomb data.
All xG and goals are open play only; league tables use full match results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xg.data import OpenData  # noqa: E402
from xg.tables import finishing_table, team_season_table  # noqa: E402

LEAGUES = ["Premier League", "La Liga", "Serie A", "Ligue 1"]


def md(df: pd.DataFrame, floatfmt: str = "{:.1f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            elif isinstance(v, (int,)) or (hasattr(v, "dtype") and str(getattr(v, "dtype", "")).startswith("int")):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def finishing_md(t: pd.DataFrame) -> str:
    t = t.reset_index().rename(columns={"player name": "Player", "shots": "Shots", "goals": "Goals",
                                        "xg": "xG", "goals_minus_xg": "Goals − xG", "z": "z"})
    t["z"] = t["z"].map("{:+.1f}".format)
    t["Goals − xG"] = t["Goals − xG"].map("{:+.1f}".format)
    return md(t)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--oof", default="results/oof_xg.parquet")
    p.add_argument("--out", default="docs/TABLES.md")
    p.add_argument("--csv-dir", default="results")
    args = p.parse_args(argv)
    oof = pd.read_parquet(args.oof)
    od = OpenData()
    parts = []

    # ---- 1. 2015/16 league tables with xG
    summary = []
    for comp in LEAGUES:
        t = team_season_table(oof, od, comp, "2015/2016")
        t.to_csv(Path(args.csv_dir) / f"table_{comp.replace(' ', '_').lower()}_2015_16.csv")
        rho_xg = spearmanr(t["op_xg_diff"], t["points"]).statistic
        rho_g = spearmanr(t["op_goals"] - t["op_goals_against"], t["points"]).statistic
        summary.append({"League": comp, "Matches": int(t["played"].sum() // 2),
                        "Spearman ρ (open-play xG diff vs points)": f"{rho_xg:.2f}",
                        "Spearman ρ (open-play goal diff vs points)": f"{rho_g:.2f}"})
        show = t.reset_index().rename(columns={"team": "Team"})
        show = pd.DataFrame({
            "Pos": show["position"], "Team": show["Team"], "Pts": show["points"],
            "Goals for (OP)": show["op_goals"], "xG for": show["op_xg"],
            "Goals against (OP)": show["op_goals_against"], "xG against": show["op_xg_against"],
            "xG diff": show["op_xg_diff"], "Rank by xG diff": show["xg_diff_rank"],
        })
        note = (" StatsBomb's data is missing 3 of the 380 matches, so points and positions differ slightly"
                " from the official table (PSG finished on 96 points)." if comp == "Ligue 1" else
                " Points and positions match the official final table.")
        parts.append(f"### {comp} 2015/16\n\nOP = open play.{note}\n\n" + md(show))
    league_intro = md(pd.DataFrame(summary))

    # ---- 2. player finishing, careers
    car = finishing_table(oof, ["player name"], min_shots=150)
    car.to_csv(Path(args.csv_dir) / "finishing_players.csv")
    over, under = car.head(10), car.tail(5).iloc[::-1]

    # ---- 3. Barcelona by season (La Liga)
    b = oof[oof["competition"].eq("La Liga") & oof["Team Name"].eq("Barcelona")]
    bs = finishing_table(b, ["season_id"], min_shots=1).sort_index().reset_index()
    bs = bs.rename(columns={"season_id": "Season", "shots": "Shots", "goals": "Goals", "xg": "xG",
                            "goals_minus_xg": "Goals − xG", "z": "z"})
    bs["Goals ÷ xG"] = (bs["Goals"] / bs["xG"]).map("{:.2f}".format)
    bs["z"] = bs["z"].map("{:+.1f}".format)

    # ---- 4. players named in the thesis
    named = ["Lionel Andrés Messi Cuccittini", "Luis Alberto Suárez Díaz", "Neymar da Silva Santos Junior",
             "Andrés Iniesta Luján"]
    th = finishing_table(oof[oof["player name"].isin(named)], ["player name"], min_shots=1)

    doc = f"""# Team and player tables (out-of-fold xG)

Card #6. Every shot's xG here comes from a model that **never saw that shot's match**: 5-fold cross-validation grouped by match, using the best model from [RESULTS.md](RESULTS.md). These tables replace the ones at the end of `EDA+Model.ipynb`, which counted only the random 20% test sample, covered only Barcelona's matches, and used a hand-typed list of league positions.

- **Open play only:** xG and goals cover open-play shots, the shots the model covers. Penalties, direct free kicks and own goals are not included, so "goals" here is lower than a team's or player's official total.
- **Real league tables:** points and positions are calculated from StatsBomb's match results, not typed in.
- **Reading z:** *z* is the gap between goals and xG divided by how much that gap would vary by chance for an average finisher taking the same shots. Roughly, |z| above 2 is unlikely to be luck alone.

```bash
python scripts/xg_tables.py --oof results/oof_xg.parquet --out docs/TABLES.md
```

## 1. League tables 2015/16, with xG

How well each measure tracks final points (rank correlation; 1.00 would be perfect):

{league_intro}

Within a single season, actual goal difference tracks points more closely than xG difference. That is expected, because points are decided by actual goals. xG measures the quality of the chances a team created and allowed, and the gap between the two shows who did better or worse than their chances suggest. In 2015/16, Leicester won the Premier League while ranking 6th on open-play xG difference, and Arsenal, 1st on xG difference, finished second. Ligue 1's lower correlations partly reflect the 3 missing matches.

{chr(10).join(parts)}

## 2. Finishing: goals vs xG over a career (open play, at least 150 shots)

Biggest over-performers:

{finishing_md(over)}

Biggest under-performers:

{finishing_md(under)}

Data coverage varies by player: Barcelona players appear in every Barcelona La Liga match from 2004/05 to 2020/21, while most other players appear only in the competitions listed in [DATA.md](DATA.md).

## 3. Barcelona in La Liga, season by season (open play)

{md(bs[["Season", "Shots", "Goals", "xG", "Goals − xG", "Goals ÷ xG", "z"]])}

## 4. Players named in the thesis

{finishing_md(th)}
"""
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

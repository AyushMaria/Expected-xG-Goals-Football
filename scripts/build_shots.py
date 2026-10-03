"""Build a shot table from StatsBomb open data.

Examples (run from the repository root):

    # Reproduce the thesis dataset: every La Liga match Barcelona played, 2004/05-2020/21
    python scripts/build_shots.py --competition "La Liga" --team Barcelona \
        --seasons 2004/2005-2020/2021 --out data/shots_laliga_barcelona.parquet

    # Every men's match in the open data
    python scripts/build_shots.py --out data/shots_all_men.parquet

Raw match files are cached in data/raw/ (or $XG_CACHE_DIR), so re-runs are fast.
Data source: StatsBomb open data (https://github.com/statsbomb/open-data).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xg.data import OpenData, select_matches  # noqa: E402
from xg.features import build_shot_table  # noqa: E402


def season_range(spec: str) -> list[str]:
    """'2004/2005-2020/2021' -> ['2004/2005', ..., '2020/2021']; '2015/2016' -> ['2015/2016']."""
    if "-" not in spec:
        return [spec]
    start, end = spec.split("-")
    first, last = int(start.split("/")[0]), int(end.split("/")[0])
    return [f"{y}/{y + 1}" for y in range(first, last + 1)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--competition", help='Competition name, e.g. "La Liga". Default: all competitions.')
    p.add_argument("--seasons", help='Season or range, e.g. "2015/2016" or "2004/2005-2020/2021". Default: all.')
    p.add_argument("--team", help='Only matches involving this team, e.g. "Barcelona".')
    p.add_argument("--gender", default="male", choices=["male", "female"])
    p.add_argument("--out", required=True, help="Output file (.parquet or .csv).")
    p.add_argument("--cache-dir", default=None, help="Where raw JSON files are cached.")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    od = OpenData(args.cache_dir) if args.cache_dir else OpenData()
    seasons = season_range(args.seasons) if args.seasons else None
    t0 = time.time()
    matches = select_matches(od, args.competition, seasons, args.team, args.gender)
    if not matches:
        print("No matches found for that selection.", file=sys.stderr)
        return 1
    print(f"Selected {len(matches)} matches; downloading/caching events ...")
    shots = build_shot_table(od, matches, workers=args.workers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        shots.to_csv(out, index=False)
    else:
        shots.to_parquet(out, index=False)
    print(f"Wrote {len(shots):,} shots from {shots['game_id'].nunique()} matches to {out} "
          f"in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download and cache StatsBomb open data.

Reads the raw JSON files published at https://github.com/statsbomb/open-data
(the same files the ``statsbombpy`` package reads for open data) and keeps a
gzipped copy of each one in a local cache, so every file is downloaded once.

Data source: StatsBomb. If you publish work based on this data, credit
StatsBomb and use their logo, as required by the open-data terms.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_CACHE = Path(os.environ.get("XG_CACHE_DIR", Path(__file__).resolve().parents[1] / "data" / "raw"))


class OpenData:
    """Cached access to StatsBomb open data.

    Example:
        >>> od = OpenData()
        >>> comps = od.competitions()
        >>> matches = od.matches(competition_id=11, season_id=90)
        >>> events = od.events(matches[0]["match_id"])
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, retries: int = 3):
        self.cache_dir = Path(cache_dir)
        self.retries = retries

    # ------------------------------------------------------------------ fetch
    def _get(self, relpath: str):
        """Return parsed JSON for ``relpath`` (e.g. 'events/123.json'), using the cache."""
        cached = self._download(relpath)
        with gzip.open(cached, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def _download(self, relpath: str) -> Path:
        """Make sure ``relpath`` is in the cache and return the cached file's path (without parsing it)."""
        cached = self.cache_dir / (relpath + ".gz")
        if cached.exists():
            return cached
        url = f"{BASE_URL}/{relpath}"
        last_err = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    raw = resp.read()
                break
            except (urllib.error.URLError, TimeoutError) as err:  # network hiccup: back off and retry
                last_err = err
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Could not download {url}: {last_err}")
        json.loads(raw)  # fail now, not later, if the download is not valid JSON
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(cached.name + f".{os.getpid()}.tmp")
        with gzip.open(tmp, "wb") as fh:
            fh.write(raw)
        tmp.replace(cached)  # atomic: a half-written file never looks cached
        return cached

    # ------------------------------------------------------------------ API
    def competitions(self) -> list[dict]:
        """All competition-seasons in the open data (one dict per competition-season)."""
        return self._get("competitions.json")

    def matches(self, competition_id: int, season_id: int) -> list[dict]:
        """All matches for one competition-season."""
        return self._get(f"matches/{competition_id}/{season_id}.json")

    def events(self, match_id: int) -> list[dict]:
        """All events for one match, in StatsBomb's raw JSON format."""
        return self._get(f"events/{match_id}.json")

    def prefetch_events(self, match_ids: Iterable[int], workers: int = 8) -> None:
        """Download (and cache) the events of many matches in parallel, without loading them into memory."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(lambda mid: self._download(f"events/{mid}.json"), match_ids):
                pass


def select_matches(
    od: OpenData,
    competition_name: str | None = None,
    season_names: Iterable[str] | None = None,
    team: str | None = None,
    gender: str = "male",
) -> list[dict]:
    """Pick matches from the open data.

    Args:
        competition_name: e.g. "La Liga". ``None`` means every competition.
        season_names: e.g. ["2015/2016"]. ``None`` means every season.
        team: keep only matches involving this team, e.g. "Barcelona".
        gender: "male" or "female" (StatsBomb's ``competition_gender``).

    Each returned match dict gets two extra keys, ``competition_name`` and
    ``season_name``, so shots can be labelled later.
    """
    seasons = set(season_names) if season_names is not None else None
    selected = []
    for comp in od.competitions():
        if comp.get("competition_gender") != gender:
            continue
        if competition_name is not None and comp["competition_name"] != competition_name:
            continue
        if seasons is not None and comp["season_name"] not in seasons:
            continue
        for m in od.matches(comp["competition_id"], comp["season_id"]):
            if team is not None and team not in (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]):
                continue
            m = dict(m)
            m["competition_name"] = comp["competition_name"]
            m["season_name"] = comp["season_name"]
            selected.append(m)
    return selected

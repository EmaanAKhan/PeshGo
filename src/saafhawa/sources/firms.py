"""NASA FIRMS: satellite fire detections (crop burning) upwind of Lahore.

Needs a free MAP_KEY in .env (FIRMS_MAP_KEY), from
https://firms.modaps.eosdis.nasa.gov/api/map_key/ . The area API returns at
most 10 days per call, so history is fetched in 10-day windows.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from .. import manifest
from ..settings import config, secrets
from .http import FetchError, get

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def _key() -> str:
    k = secrets().firms_key
    if not k:
        raise FetchError("FIRMS_MAP_KEY is not set. Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and add it to .env")
    return k


def _area() -> str:
    w, s, e, n = config()["firms"]["bbox"]
    return f"{w},{s},{e},{n}"


LAST_FAILURES: list[str] = []


def _name(source: str, start: date, days: int, tag: str) -> str:
    return f"firms/{tag}/{source}_{start}_{days}d.csv"


def _fetch(source: str, start: date, days: int, tag: str) -> bool:
    url = f"{BASE}/{_key()}/{source}/{_area()}/{days}/{start.isoformat()}"
    try:
        r = get(url, key="firms", per_minute=60)
    except FetchError as e:
        raise FetchError(str(e).replace(_key(), "<MAP_KEY>")) from None  # never print the key
    text = r.content
    if text.lstrip().lower().startswith((b"invalid", b"error")):
        raise FetchError(f"FIRMS: {text[:200]!r}")
    return manifest.save("nasa-firms", f"{source} bbox={_area()} from {start} days={days}", _name(source, start, days, tag), text).new


MAX_DAYS = 5  # NASA's area API rejects more ("Invalid day range. Expects [1..5]"); 10 made every history request fail


def fetch_history(until: date | None = None) -> int:
    """Windows of up to 5 days from history_start; finished archive windows are skipped on
    the next run, a few windows are fetched at once, and one failing window does not stop the rest."""
    cfg = config()
    fc = cfg["firms"]
    days = max(1, min(int(fc["day_range"]), MAX_DAYS))
    d = date.fromisoformat(cfg["history_start"])
    end = until or (datetime.now(timezone.utc).date() - timedelta(days=60))
    LAST_FAILURES.clear()
    # The standard archive lags about two months; near-real-time covers the gap.
    today = datetime.now(timezone.utc).date()
    todo = []
    while d <= today:
        source = fc["archive_source"] if d <= end else fc["recent_source"]
        if not (source == fc["archive_source"] and manifest.has(_name(source, d, days, "history"))):
            todo.append((source, d))
        d += timedelta(days=days)

    def one(task) -> int:
        source, start = task
        try:
            return int(_fetch(source, start, days, "history"))
        except FetchError as e:
            LAST_FAILURES.append(f"{source} {start}: {e}"[:200])
            return 0

    # The shared rate limiter in http.get keeps the total under the per-minute limit.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=fc.get("workers", 4)) as pool:
        return sum(pool.map(one, todo))


def fetch_recent() -> int:
    fc = config()["firms"]
    today = datetime.now(timezone.utc).date()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H")
    return int(_fetch(fc["recent_source"], today - timedelta(days=2), 3, f"recent/{stamp}"))

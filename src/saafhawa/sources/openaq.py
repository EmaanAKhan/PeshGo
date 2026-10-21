"""OpenAQ v3: ground PM2.5 monitors around Lahore (the target).

Needs a free API key in .env (OPENAQ_API_KEY). Locations are discovered within
the configured radius; each PM2.5 sensor's hourly averages are paged from
/v3/sensors/{id}/hours and stored page by page, exactly as received.

History is fetched in windows of `openaq.chunk_days` days (OpenAQ times out,
HTTP 408, on multi-year requests). A window that still times out is split in
half. Finished windows are skipped on the next run, so an interrupted download
resumes where it stopped. One failing sensor never stops the others: failures
are collected in LAST_FAILURES and reported by the caller.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .. import manifest
from ..settings import config, secrets
from .http import FetchError, get

BASE = "https://api.openaq.org/v3"
LAST_FAILURES: list[str] = []
MIN_WINDOW = timedelta(days=7)


def _headers() -> dict:
    key = secrets().openaq_key
    if not key:
        raise FetchError("OPENAQ_API_KEY is not set. Get a free key at https://explore.openaq.org and add it to .env")
    return {"X-API-Key": key, "Accept": "application/json"}


def _get(path: str, params: dict):
    cfg = config()["openaq"]
    return get(f"{BASE}{path}", params=params, headers=_headers(), per_minute=cfg["requests_per_minute"], key="openaq")


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def discover() -> list[dict]:
    """PM2.5 sensors near the city centre. Saves the raw location listing."""
    cfg = config()
    oq = cfg["openaq"]
    params = {"coordinates": f"{cfg['center']['lat']},{cfg['center']['lon']}", "radius": oq["radius_m"], "limit": 1000}
    if oq["location_ids"]:
        params = {"limit": 1000}
    r = _get("/locations", params)
    manifest.save("openaq", f"locations near {cfg['city']} r={oq['radius_m']}m", f"openaq/locations_{datetime.now(timezone.utc):%Y%m%dT%H}.json", r.content)
    return sensors_from_locations(r.json(), oq)


def sensors_from_locations(payload: dict, oq: dict) -> list[dict]:
    wanted = set(oq.get("location_ids") or [])
    out = []
    for loc in payload.get("results", []):
        if wanted and loc.get("id") not in wanted:
            continue
        for s in loc.get("sensors", []):
            param = (s.get("parameter") or {}).get("name")
            if param != oq["parameter"]:
                continue
            out.append({
                "location_id": loc.get("id"),
                "location": loc.get("name"),
                "sensor_id": s.get("id"),
                "units": (s.get("parameter") or {}).get("units"),
                "is_monitor": bool(loc.get("isMonitor")),
                "provider": (loc.get("provider") or {}).get("name"),
                "lat": (loc.get("coordinates") or {}).get("latitude"),
                "lon": (loc.get("coordinates") or {}).get("longitude"),
                "first": ((loc.get("datetimeFirst") or {}).get("utc")),
                "last": ((loc.get("datetimeLast") or {}).get("utc")),
            })
    return out


def _window_name(sensor: dict, start: datetime, end: datetime, tag: str, page: int) -> str:
    return f"openaq/{tag}/sensor_{sensor['sensor_id']}_{start:%Y%m%dT%H}_{end:%Y%m%dT%H}_p{page:03d}.json"


def fetch_sensor_hours(sensor: dict, start: datetime, end: datetime, tag: str) -> int:
    """Page through one sensor's hourly values between start and end (UTC)."""
    oq = config()["openaq"]
    page, n = 1, 0
    while True:
        params = {
            "datetime_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": oq["page_limit"],
            "page": page,
        }
        r = _get(f"/sensors/{sensor['sensor_id']}/hours", params)
        body = r.json()
        # Keep the sensor's metadata next to its values: the raw file must explain itself.
        wrapped = {"sensor": sensor, "request": params, "response": body}
        saved = manifest.save("openaq", f"sensor {sensor['sensor_id']} hours {params['datetime_from']}..{params['datetime_to']} page {page}",
                              _window_name(sensor, start, end, tag, page),
                              json.dumps(wrapped, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        n += saved.new
        results = body.get("results", [])
        if len(results) < oq["page_limit"]:
            return n
        page += 1


def _fetch_window(sensor: dict, start: datetime, end: datetime, tag: str) -> int:
    """One window; on a timeout (HTTP 408) it is split in half, down to a week."""
    try:
        return fetch_sensor_hours(sensor, start, end, tag)
    except FetchError as e:
        if "HTTP 408" in str(e) and end - start > MIN_WINDOW:
            mid = (start + (end - start) / 2).replace(minute=0, second=0, microsecond=0)
            return _fetch_window(sensor, start, mid, tag) + _fetch_window(sensor, mid, end, tag)
        raise


def _active_between(sensor: dict, start: datetime, end: datetime) -> bool:
    first, last = _ts(sensor.get("first")), _ts(sensor.get("last"))
    return not ((last and last < start) or (first and first > end))


def fetch_history() -> int:
    """All sensors, from history_start to two days ago, in resumable windows."""
    cfg = config()
    days = cfg["openaq"].get("chunk_days", 90)
    start0 = datetime.fromisoformat(cfg["history_start"]).replace(tzinfo=timezone.utc)
    end0 = (datetime.now(timezone.utc) - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    LAST_FAILURES.clear()
    only = set(cfg["openaq"].get("history_sensor_ids") or [])
    sensors = [s for s in discover() if _active_between(s, start0, end0) and (not only or s["sensor_id"] in only)]
    print(f"   OpenAQ: {len(sensors)} PM2.5 sensors with data since {start0:%Y-%m-%d}", flush=True)

    def one(s):
        got = 0
        # Windows are aligned to history_start, so their names are the same on every run.
        w = start0
        while w < end0:
            e = min(w + timedelta(days=days), end0)
            full = e - w == timedelta(days=days)
            if _active_between(s, w, e) and not (full and manifest.has(_window_name(s, w, e, "history", 1))):
                try:
                    got += _fetch_window(s, w, e, "history")
                except FetchError as err:
                    LAST_FAILURES.append(f"sensor {s['sensor_id']} {w:%Y-%m-%d}..{e:%Y-%m-%d}: {err}"[:200])
            w = e
        return s, got

    # The API answers slowly, so a few sensors are fetched at once; the shared
    # rate limiter in http.get keeps the total under requests_per_minute.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = 0
    with ThreadPoolExecutor(max_workers=cfg["openaq"].get("workers", 4)) as pool:
        for i, fut in enumerate(as_completed([pool.submit(one, s) for s in sensors]), 1):
            s, got = fut.result()
            n += got
            print(f"   OpenAQ sensor {i}/{len(sensors)} ({s['sensor_id']}, {s.get('location') or ''}): {got} new files", flush=True)
    return n


def fetch_recent(hours: int = 48) -> int:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours)
    stamp = end.strftime("%Y%m%dT%H")
    LAST_FAILURES.clear()
    n, tried = 0, 0
    for s in discover():
        # Sensors silent for more than a week are not asked every hour.
        last = _ts(s.get("last"))
        if last and last < end - timedelta(days=7):
            continue
        tried += 1
        try:
            n += fetch_sensor_hours(s, start, end, tag=f"recent/{stamp}")
        except FetchError as err:
            LAST_FAILURES.append(f"sensor {s['sensor_id']}: {err}"[:200])
    if tried and len(LAST_FAILURES) == tried:
        raise FetchError(f"every OpenAQ sensor failed; first: {LAST_FAILURES[0]}")
    return n

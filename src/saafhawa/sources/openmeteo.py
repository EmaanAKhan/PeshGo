"""Open-Meteo: hourly weather (ERA5 archive + forecast) and CAMS PM2.5.

No key is needed. History is fetched in chunks; the hourly pipeline fetches
the last few days plus the forecast, so new hours and revisions both arrive.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from .. import manifest
from ..settings import config
from .http import get

WEATHER_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST = "https://api.open-meteo.com/v1/forecast"
AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"


def _chunks(start: date, end: date, days: int):
    d = start
    while d <= end:
        e = min(d + timedelta(days=days - 1), end)
        yield d, e
        d = e + timedelta(days=1)


def _params(cfg, variables, extra):
    return {
        "latitude": cfg["center"]["lat"],
        "longitude": cfg["center"]["lon"],
        "hourly": ",".join(variables),
        "timezone": "GMT",
        **extra,
    }


def fetch_history(until: date | None = None) -> int:
    """Weather archive and CAMS archive from history_start to `until`. Returns files written."""
    cfg = config()
    om = cfg["openmeteo"]
    start = date.fromisoformat(cfg["history_start"])
    # The ERA5 archive lags a few days; the forecast endpoint covers the rest.
    end = until or (datetime.now(timezone.utc).date() - timedelta(days=6))
    n = 0
    for s, e in _chunks(start, end, om["chunk_days"]):
        # Finished chunks are not downloaded again (resumes an interrupted run).
        if (e - s).days == om["chunk_days"] - 1 and manifest.has(f"open_meteo/weather_{s}_{e}.json") and manifest.has(f"open_meteo/air_{s}_{e}.json"):
            continue
        q = {"start_date": s.isoformat(), "end_date": e.isoformat()}
        r = get(WEATHER_ARCHIVE, params=_params(cfg, om["weather_vars"], q), key="open-meteo")
        n += manifest.save("open-meteo-weather", f"archive {cfg['city']} {s}..{e}", f"open_meteo/weather_{s}_{e}.json", r.content).new
        r = get(AIR, params=_params(cfg, om["air_vars"], q), key="open-meteo")
        n += manifest.save("open-meteo-air", f"CAMS {cfg['city']} {s}..{e}", f"open_meteo/air_{s}_{e}.json", r.content).new
    return n


def fetch_recent(past_days: int = 7, forecast_days: int = 2) -> int:
    """Latest hours and the next days' forecast (weather and CAMS), stamped by fetch hour."""
    cfg = config()
    om = cfg["openmeteo"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H")
    q = {"past_days": past_days, "forecast_days": forecast_days}
    n = 0
    r = get(WEATHER_FORECAST, params=_params(cfg, om["weather_vars"], q), key="open-meteo")
    n += manifest.save("open-meteo-weather", f"forecast {cfg['city']} past={past_days} next={forecast_days}", f"open_meteo/recent/weather_{stamp}.json", r.content).new
    r = get(AIR, params=_params(cfg, om["air_vars"], q), key="open-meteo")
    n += manifest.save("open-meteo-air", f"CAMS forecast {cfg['city']} past={past_days} next={forecast_days}", f"open_meteo/recent/air_{stamp}.json", r.content).new
    return n

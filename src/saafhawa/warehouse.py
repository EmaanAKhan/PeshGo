"""Raw -> cleaned -> gold, in DuckDB.

raw      files exactly as fetched (data/raw, listed in data/manifest.csv)
cleaned  parsed, typed, UTC, de-duplicated (tables clean_*)
gold     quality-checked tables with stable row IDs (tables gold_*); every
         number the dashboard or the Insight Agent shows comes from here.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import manifest
from .settings import config, path


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path("warehouse")), read_only=read_only)
    # Timestamps come back in UTC whatever the machine's own time zone is.
    con.execute("SET TimeZone = 'UTC'")
    return con


# ── raw -> cleaned ──────────────────────────────────────────────────────────

def _fetch_times() -> dict[str, str]:
    return {r["raw_path"]: r["fetch_time"] for r in manifest.read()}


def _open_meteo(source: str) -> pd.DataFrame:
    """All Open-Meteo files of one source, one row per hour.

    For an hour present in several files, the archive (reanalysis) value wins
    over a forecast-run value, and among forecast runs the latest fetch wins.
    """
    ft = _fetch_times()
    frames = []
    for p in manifest.files(source):
        body = json.loads(p.read_text(encoding="utf-8"))
        if body.get("timezone") not in ("GMT", "UTC"):
            raise ValueError(f"{p}: expected GMT timestamps, got {body.get('timezone')}")
        hourly = body.get("hourly") or {}
        df = pd.DataFrame(hourly)
        if df.empty:
            continue
        df["ts_utc"] = pd.to_datetime(df.pop("time"), utc=True)
        rel = p.relative_to(path("data")).as_posix()
        df["is_archive"] = "/recent/" not in rel
        df["fetch_time"] = ft.get(rel, "")
        df["raw_path"] = rel
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ts_utc", "is_archive", "fetch_time"]).drop_duplicates("ts_utc", keep="last")
    return df.reset_index(drop=True)


def clean_weather() -> pd.DataFrame:
    df = _open_meteo("open-meteo-weather")
    if df.empty:
        return df
    return df.rename(columns={
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "rh_pct",
        "wind_speed_10m": "wind_kmh",
        "wind_direction_10m": "wind_from_deg",
        "boundary_layer_height": "blh_m",
        "precipitation": "precip_mm",
    })


def clean_cams() -> pd.DataFrame:
    df = _open_meteo("open-meteo-air")
    if df.empty:
        return df
    return df.rename(columns={"pm2_5": "cams_pm25"})[["ts_utc", "cams_pm25", "is_archive", "fetch_time", "raw_path"]]


def clean_pm25() -> tuple[pd.DataFrame, pd.DataFrame]:
    """OpenAQ hourly values (all sensors) and the sensor table."""
    ft = _fetch_times()
    rows, sensors = [], {}
    for p in manifest.files("openaq"):
        body = json.loads(p.read_text(encoding="utf-8"))
        if "sensor" not in body:  # a locations listing
            continue
        s = body["sensor"]
        sensors[s["sensor_id"]] = s
        rel = p.relative_to(path("data")).as_posix()
        for r in body["response"].get("results", []):
            period = r.get("period") or {}
            start = ((period.get("datetimeFrom") or {}).get("utc"))
            if start is None:
                continue
            rows.append({
                "sensor_id": s["sensor_id"],
                "location_id": s["location_id"],
                "ts_utc": start,
                "value_raw": r.get("value"),
                "units": (r.get("parameter") or {}).get("units") or s.get("units"),
                "coverage_pct": ((r.get("coverage") or {}).get("percentComplete")),
                "fetch_time": ft.get(rel, ""),
                "raw_path": rel,
            })
    df = pd.DataFrame(rows)
    st = pd.DataFrame(list(sensors.values()))
    if df.empty:
        return df, st
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True).dt.floor("h")
    return df, st


def clean_fires() -> pd.DataFrame:
    frames = []
    for p in manifest.files("nasa-firms"):
        text = p.read_text(encoding="utf-8").strip()
        if not text or "\n" not in text:
            continue
        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            continue
        df["raw_path"] = p.relative_to(path("data")).as_posix()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts_utc", "lat", "lon", "frp", "confidence", "satellite"])
    df = pd.concat(frames, ignore_index=True)
    t = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["ts_utc"] = pd.to_datetime(df["acq_date"] + " " + t.str[:2] + ":" + t.str[2:], utc=True)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df = df.drop_duplicates(["ts_utc", "lat", "lon", "satellite"])
    return df[["ts_utc", "lat", "lon", "frp", "confidence", "satellite", "raw_path"]].reset_index(drop=True)


# ── build ───────────────────────────────────────────────────────────────────

def build(run_id: str) -> dict:
    """Rebuild cleaned and gold layers from the raw files. Returns a summary."""
    from .quality_agent import QualityAgent

    pm, sensors = clean_pm25()
    weather = clean_weather()
    cams = clean_cams()
    fires = clean_fires()

    agent = QualityAgent(run_id)
    weather = agent.check_weather(weather)
    cams = agent.check_cams(cams)
    pm_gold, sensors = agent.check_pm25(pm, sensors)
    pm_gold = agent.apply_humidity(pm_gold, weather)
    agent.check_freshness(pm_gold, weather, cams, fires)

    con = connect()
    try:
        for name, df in {
            "clean_pm25": pm, "clean_weather": weather, "clean_cams": cams, "clean_fires": fires,
            "sensors": sensors, "gold_pm25_hourly": pm_gold,
        }.items():
            con.register("_df", df if not df.empty else pd.DataFrame({"_empty": []}))
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
            con.unregister("_df")
        agent.write(con)
        from .features import build_features
        feats = build_features(pm_gold, weather, cams, fires, sensors)
        con.register("_df", feats)
        con.execute("CREATE OR REPLACE TABLE gold_features AS SELECT * FROM _df")
        con.unregister("_df")
        agent.check_drift(con, feats)
        agent.write(con)
    finally:
        con.close()
    return {
        "pm25_rows": int(len(pm)),
        "gold_pm25_rows": int(pm_gold["usable"].sum()) if len(pm_gold) else 0,
        "sensors": int(len(sensors)),
        "weather_hours": int(len(weather)),
        "cams_hours": int(len(cams)),
        "fires": int(len(fires)),
        "feature_rows": int(len(feats)),
        "quality_decisions": agent.count,
    }


def table(name: str, query: str | None = None) -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute(query or f"SELECT * FROM {name}").df()
    finally:
        con.close()

"""Model features and targets (gold_features), one row per sensor and issue hour.

Leakage rule: every feature at issue time t uses information from t or
earlier, with one documented exception, `cams_next24`: the CAMS forecast for
the next 24 hours, which is published ahead of time and is available at t
in live use. The target is the mean PM2.5 over hours t+1 .. t+24.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import config

FEATURES = [
    "pm25_now", "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12", "pm25_lag24",
    "pm25_mean6", "pm25_mean24", "pm25_mean72", "pm25_std24", "pm25_max24",
    "city_median_now", "city_mean24",
    "cams_now", "cams_prev24", "cams_next24",
    "temp_c", "rh_pct", "wind_kmh", "wind_sin", "wind_cos", "blh_m", "blh_mean6", "wind_mean6", "precip_24",
    "fires_48h", "fires_upwind_48h", "frp_upwind_48h",
    "hour_pkt", "dow", "month_sin", "month_cos", "smog_season",
    "is_low_cost", "humid_bias_risk",
]
# No sensor-identity feature: a code numbered by sorting "the sensors that exist today" changes whenever a
# monitor is added (the model then reads another sensor's code), and most monitors are unseen at training time.
CATEGORICAL: list[str] = []


def _bearing(lat1, lon1, lat2, lon2):
    """Bearing in degrees from point 1 to point 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def fire_features(fires: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Per hour: fires in the box in the last 48 h, and those upwind of the city."""
    cfg = config()
    hours = weather[["ts_utc", "wind_from_deg"]].copy() if len(weather) else pd.DataFrame(columns=["ts_utc", "wind_from_deg"])
    if fires.empty or hours.empty:
        return hours.assign(fires_48h=0.0, fires_upwind_48h=0.0, frp_upwind_48h=0.0)[["ts_utc", "fires_48h", "fires_upwind_48h", "frp_upwind_48h"]]
    c = cfg["center"]
    f = fires.copy()
    f["bearing"] = _bearing(c["lat"], c["lon"], f["lat"], f["lon"])
    f["hour"] = f["ts_utc"].dt.floor("h")
    half = cfg["firms"]["upwind_cone_deg"] / 2
    # Bin fires by hour and by 10-degree bearing sector, then roll a 48-hour window.
    f["sector"] = (f["bearing"] // 10).astype(int)
    grid = pd.date_range(hours["ts_utc"].min(), hours["ts_utc"].max(), freq="h", tz="UTC")
    counts = f.pivot_table(index="hour", columns="sector", values="frp", aggfunc="size", fill_value=0).reindex(grid, fill_value=0)
    frp = f.pivot_table(index="hour", columns="sector", values="frp", aggfunc="sum", fill_value=0).reindex(grid, fill_value=0)
    for m in (counts, frp):
        for s in range(36):
            if s not in m:
                m[s] = 0
    counts = counts[sorted(counts.columns)].rolling(48, min_periods=1).sum()
    frp = frp[sorted(frp.columns)].rolling(48, min_periods=1).sum()
    centres = np.arange(36) * 10 + 5
    wd = hours.set_index("ts_utc")["wind_from_deg"].reindex(grid)
    diff = np.abs(((centres[None, :] - wd.to_numpy()[:, None]) + 180) % 360 - 180)
    upwind = diff <= half
    out = pd.DataFrame({
        "ts_utc": grid,
        "fires_48h": counts.to_numpy().sum(axis=1),
        "fires_upwind_48h": np.where(np.isnan(wd.to_numpy())[:, None], 0, counts.to_numpy() * upwind).sum(axis=1),
        "frp_upwind_48h": np.where(np.isnan(wd.to_numpy())[:, None], 0, frp.to_numpy() * upwind).sum(axis=1),
    })
    return out


def build_features(pm: pd.DataFrame, weather: pd.DataFrame, cams: pd.DataFrame, fires: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    cfg = config()
    h = cfg["model"]["horizon_hours"]
    need = cfg["model"]["min_target_hours"]
    if pm.empty:
        return pd.DataFrame(columns=["row_id", "location_id", "sensor_id", "issued_utc", "target_next24", "baseline_persistence", *FEATURES])

    use = pm.copy()
    use.loc[~use["usable"].astype(bool), "pm25"] = np.nan

    # City-wide signal: median across sensors each hour.
    city = use.groupby("ts_utc")["pm25"].median().rename("city_median_now")
    city_grid = pd.date_range(city.index.min(), city.index.max(), freq="h", tz="UTC")
    city = city.reindex(city_grid)
    city_mean24 = city.rolling(24, min_periods=12).mean().rename("city_mean24")

    w = weather.set_index("ts_utc").sort_index() if len(weather) else pd.DataFrame()
    cm = cams.set_index("ts_utc")["cams_pm25"].sort_index() if len(cams) else pd.Series(dtype=float)
    ff = fire_features(fires, weather).set_index("ts_utc") if len(weather) else pd.DataFrame()

    frames = []
    for sid, g in use.groupby("sensor_id"):
        g = g.set_index("ts_utc").sort_index()
        s = g["pm25"]
        d = pd.DataFrame(index=g.index)
        d["row_id"] = "feat:" + str(int(sid)) + ":" + g.index.strftime("%Y%m%d%H")
        d["sensor_id"] = int(sid)
        d["location_id"] = g["location_id"].astype("Int64")
        d["pm25_now"] = s
        for k in (1, 3, 6, 12, 24):
            d[f"pm25_lag{k}"] = s.shift(k)
        d["pm25_mean6"] = s.rolling(6, min_periods=4).mean()
        d["pm25_mean24"] = s.rolling(24, min_periods=need).mean()
        d["pm25_mean72"] = s.rolling(72, min_periods=48).mean()
        d["pm25_std24"] = s.rolling(24, min_periods=need).std()
        d["pm25_max24"] = s.rolling(24, min_periods=need).max()
        # Declared baseline: next 24 h = the last 24 h.
        d["baseline_persistence"] = d["pm25_mean24"]
        # Target: mean of t+1 .. t+h, with at least `need` observed hours.
        fut = s[::-1].rolling(h, min_periods=need).mean()[::-1].shift(-1)
        d["target_next24"] = fut
        d["target_hours"] = s[::-1].rolling(h, min_periods=1).count()[::-1].shift(-1)
        d.loc[d["target_hours"] < need, "target_next24"] = np.nan
        d["is_low_cost"] = g["is_low_cost"].astype(float)
        d["humid_bias_risk"] = g["humid_bias_risk"].astype(float) if "humid_bias_risk" in g else 0.0
        frames.append(d)
    f = pd.concat(frames).rename_axis("issued_utc").reset_index()

    f["city_median_now"] = f["issued_utc"].map(city)
    f["city_mean24"] = f["issued_utc"].map(city_mean24)

    if len(cm):
        cgrid = pd.date_range(min(cm.index.min(), f["issued_utc"].min()), max(cm.index.max(), f["issued_utc"].max()) + pd.Timedelta(hours=h), freq="h", tz="UTC")
        cs = cm[~cm.index.duplicated()].reindex(cgrid)
        f["cams_now"] = f["issued_utc"].map(cs)
        f["cams_prev24"] = f["issued_utc"].map(cs.rolling(24, min_periods=18).mean())
        nxt = cs[::-1].rolling(h, min_periods=need).mean()[::-1].shift(-1)
        f["cams_next24"] = f["issued_utc"].map(nxt)
    else:
        f[["cams_now", "cams_prev24", "cams_next24"]] = np.nan

    if len(w):
        wgrid = pd.date_range(w.index.min(), w.index.max(), freq="h", tz="UTC")
        wr = w[~w.index.duplicated()].reindex(wgrid)
        for col in ["temp_c", "rh_pct", "wind_kmh", "blh_m"]:
            f[col] = f["issued_utc"].map(wr[col])
        rad = np.radians(f["issued_utc"].map(wr["wind_from_deg"]))
        f["wind_sin"], f["wind_cos"] = np.sin(rad), np.cos(rad)
        f["blh_mean6"] = f["issued_utc"].map(wr["blh_m"].rolling(6, min_periods=3).mean())
        f["wind_mean6"] = f["issued_utc"].map(wr["wind_kmh"].rolling(6, min_periods=3).mean())
        f["precip_24"] = f["issued_utc"].map(wr["precip_mm"].rolling(24, min_periods=12).sum())
    else:
        for col in ["temp_c", "rh_pct", "wind_kmh", "blh_m", "wind_sin", "wind_cos", "blh_mean6", "wind_mean6", "precip_24"]:
            f[col] = np.nan

    if len(ff):
        for col in ["fires_48h", "fires_upwind_48h", "frp_upwind_48h"]:
            f[col] = f["issued_utc"].map(ff[col])
    else:
        f[["fires_48h", "fires_upwind_48h", "frp_upwind_48h"]] = np.nan

    pkt = f["issued_utc"].dt.tz_convert(cfg["display_timezone"])
    f["hour_pkt"] = pkt.dt.hour
    f["dow"] = pkt.dt.dayofweek
    m = pkt.dt.month
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    f["smog_season"] = m.isin([10, 11, 12, 1]).astype(int)

    # A row is only useful if its sensor reported recently.
    f = f[f["pm25_mean6"].notna()].reset_index(drop=True)
    return f[["row_id", "location_id", "sensor_id", "issued_utc", "target_next24", "target_hours", "baseline_persistence", *FEATURES]]

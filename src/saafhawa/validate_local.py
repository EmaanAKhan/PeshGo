"""Local validation against Punjab EPD's daily AQI reports.

EPD (Environment Protection Department, Government of the Punjab) publishes a
daily PDF with each district's 24-hour AQI. This downloads a limited number of
recent reports (politely, into the raw layer with the manifest), reads Lahore's
value, and compares it with the AQI computed from our own OpenAQ PM2.5 for the
same day. It checks our ground data against an official Pakistani source.

Note: EPD's report uses its own category bands; the comparison is of the
numeric AQI only, computed here with the US EPA PM2.5 formula.
"""

from __future__ import annotations

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd

from . import manifest, warehouse
from .settings import config, path
from .sources.http import get

LISTING = "https://epd.punjab.gov.pk/air_quality"
BASE = "https://epd.punjab.gov.pk"
LAHORE = {"لاہور", "رہولا", "روہال", "لاھور"}
UA = {"User-Agent": "PeshGo-hackathon/1.0 (research; local validation)"}

# US EPA PM2.5 24-hour AQI breakpoints (2024).
BREAKS = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200),
          (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]


def pm25_to_aqi(c: float) -> float:
    if c is None or np.isnan(c):
        return np.nan
    c = np.floor(c * 10) / 10
    for lo, hi, alo, ahi in BREAKS:
        if c <= hi:
            return round((ahi - alo) / (hi - lo) * (c - lo) + alo)
    return 500.0


def report_links(limit: int = 30) -> list[tuple[str, str]]:
    html = get(LISTING, headers=UA, key="epd", per_minute=10).text
    links = re.findall(r'href=["\']([^"\']*AQI%20(?:Report|Daily)%20(\d{2}\.\d{2}\.\d{4})[^"\']*\.pdf)["\']', html)
    seen, out = set(), []
    for href, d in links:
        if d in seen:
            continue
        seen.add(d)
        out.append((BASE + href if href.startswith("/") else href, d))
    out.sort(key=lambda x: datetime.strptime(x[1], "%d.%m.%Y"), reverse=True)
    return out[:limit]


def fetch_reports(limit: int = 30) -> int:
    n = 0
    for url, d in report_links(limit):
        r = get(url, headers=UA, key="epd", per_minute=10)
        n += manifest.save("punjab-epd", f"EPD daily AQI report {d} <{url}>", f"punjab_epd/aqi_{datetime.strptime(d, '%d.%m.%Y'):%Y-%m-%d}.pdf",
                           r.content, licence="Government of the Punjab public report; used for validation with attribution").new
    return n


def lahore_aqi(pdf_path) -> float | None:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    cells = [c.strip() if isinstance(c, str) else "" for c in row]
                    for i, c in enumerate(cells):
                        if c.replace(" ", "") in LAHORE:
                            nums = [x for x in cells[:i] if re.fullmatch(r"\d{2,3}", x)]
                            if nums:
                                return float(nums[-1])
    return None


def compare() -> pd.DataFrame:
    rows = []
    for f in manifest.files("punjab-epd"):
        day = f.stem.split("_", 1)[1].split("__")[0]
        rows.append({"date_pkt": day, "epd_lahore_aqi": lahore_aqi(f), "raw_path": f.relative_to(path("data")).as_posix()})
    epd = pd.DataFrame(rows)
    if epd.empty:
        return epd
    pm = warehouse.table("gold_pm25_hourly", "SELECT ts_utc, pm25 FROM gold_pm25_hourly WHERE usable")
    if len(pm):
        pm["date_pkt"] = pm["ts_utc"].dt.tz_convert(config()["display_timezone"]).dt.strftime("%Y-%m-%d")
        daily = pm.groupby("date_pkt")["pm25"].agg(["mean", "count"]).reset_index()
        epd = epd.merge(daily, on="date_pkt", how="left").rename(columns={"mean": "our_pm25_daily_mean", "count": "our_sensor_hours"})
        epd["our_aqi_us_epa"] = epd["our_pm25_daily_mean"].map(pm25_to_aqi)
        epd["difference"] = epd["our_aqi_us_epa"] - epd["epd_lahore_aqi"]
    epd = epd.sort_values("date_pkt")
    epd.to_csv(path("reports") / "local_validation.csv", index=False)
    return epd


def summary(df: pd.DataFrame) -> dict:
    if df.empty or "our_aqi_us_epa" not in df:
        return {"days": int(len(df)), "matched_days": 0}
    m = df.dropna(subset=["epd_lahore_aqi", "our_aqi_us_epa"])
    out = {"days": int(len(df)), "matched_days": int(len(m))}
    if len(m) >= 3:
        out["mean_abs_difference_aqi"] = round(float(m["difference"].abs().mean()), 1)
        out["mean_difference_aqi"] = round(float(m["difference"].mean()), 1)
        out["correlation"] = round(float(np.corrcoef(m["epd_lahore_aqi"], m["our_aqi_us_epa"])[0, 1]), 3)
    return out

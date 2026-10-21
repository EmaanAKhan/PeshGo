"""Test fixtures.

SYNTHETIC TEST DATA ONLY. Everything generated here exists to exercise the
code paths (parsing, quality rules, features, training, forecasting, the
snapshot and the Insight Agent) without network access or API keys. It is
written to a temporary folder, never to data/, and it is never used to train
or evaluate the submitted model. Reported results come only from real data
fetched by `make data`.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2024-01-01", tz="UTC")
HOURS = 152 * 24  # Jan to May 2024
SENSORS = [  # (sensor_id, location_id, is_monitor, hours of history)
    (101, 1, True, HOURS),
    (102, 2, True, HOURS),
    (103, 3, False, HOURS),
    (104, 4, True, 500),  # too short: the Quality Agent must PROPOSE dropping it
]

# Injected defects in sensor 101, by hour index. Each one maps to a rule.
TEXT_NUMBER, NEGATIVE, TOO_HIGH, DUPLICATE = 100, 200, 300, 600
STUCK = range(400, 408)       # 8 identical hours (rule: 6+)
SHORT_GAP = (500, 501)        # 2 missing hours (FILL)
LONG_GAP = range(1000, 1030)  # 30 missing hours (FLAG)


def _reset_settings():
    from saafhawa import settings
    settings.config.cache_clear()


def _series(rng, n):
    t = np.arange(n)
    season = 70 + 50 * np.cos(2 * np.pi * t / (24 * 365))  # higher in winter
    diurnal = 15 * np.cos(2 * np.pi * (t % 24 - 20) / 24)  # evening peak
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + rng.normal(0, 4)
    return np.clip(season + diurnal + ar, 3, None)


def write_fixture_raw(home: Path, seed: int = 7) -> dict:
    """Raw files in the exact formats the real sources return, saved through
    the manifest like real fetches."""
    from saafhawa import manifest

    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=HOURS, freq="h")
    iso = [t.strftime("%Y-%m-%dT%H:%M") for t in ts]
    truth = _series(rng, HOURS)
    cams = truth * 0.75 + rng.normal(0, 8, HOURS)  # CAMS reads low, as in Lahore
    t = np.arange(HOURS)
    weather = {
        "temperature_2m": (15 + 10 * np.sin(2 * np.pi * t / (24 * 365)) + 6 * np.sin(2 * np.pi * (t % 24 - 9) / 24)).round(1).tolist(),
        "relative_humidity_2m": np.clip(60 + 25 * np.sin(2 * np.pi * (t % 24 + 3) / 24) + rng.normal(0, 5, HOURS), 5, 100).round(0).tolist(),
        "wind_speed_10m": np.abs(rng.normal(7, 3, HOURS)).round(1).tolist(),
        "wind_direction_10m": rng.uniform(0, 360, HOURS).round(0).tolist(),
        "boundary_layer_height": np.clip(300 + 900 * np.sin(2 * np.pi * (t % 24 - 6) / 24), 50, None).round(0).tolist(),
        "precipitation": np.where(rng.random(HOURS) < 0.02, rng.uniform(0.1, 5, HOURS), 0).round(1).tolist(),
    }
    for chunk in range(0, HOURS, 92 * 24):
        sl = slice(chunk, chunk + 92 * 24)
        manifest.save_json("open-meteo-weather", "FIXTURE weather", f"open_meteo/weather_{chunk}.json",
                           {"timezone": "GMT", "hourly": {"time": iso[sl], **{k: v[sl] for k, v in weather.items()}}})
        manifest.save_json("open-meteo-air", "FIXTURE cams", f"open_meteo/air_{chunk}.json",
                           {"timezone": "GMT", "hourly": {"time": iso[sl], "pm2_5": cams[sl].round(1).tolist()}})

    for sid, lid, mon, n in SENSORS:
        bias = 1.0 if mon else 1.25
        vals = (truth * bias + rng.normal(0, 6, HOURS)).round(1)
        results = []
        for i in range(HOURS - n, HOURS):
            v = float(vals[i])
            if sid == 101:
                if i in LONG_GAP or i in SHORT_GAP:
                    continue
                if i == TEXT_NUMBER:
                    v = str(v)
                elif i == NEGATIVE:
                    v = -5.0
                elif i == TOO_HIGH:
                    v = 1500.0
                elif i in STUCK:
                    v = 88.8
            rec = {"value": v, "period": {"datetimeFrom": {"utc": ts[i].strftime("%Y-%m-%dT%H:%M:%SZ")}},
                   "parameter": {"units": "µg/m³"}, "coverage": {"percentComplete": 100}}
            results.append(rec)
            if sid == 101 and i == DUPLICATE:
                results.append(dict(rec))
        sensor = {"sensor_id": sid, "location_id": lid, "location": f"Fixture site {lid}", "units": "µg/m³",
                  "is_monitor": mon, "provider": "fixture", "lat": 31.52 + lid / 100, "lon": 74.35}
        manifest.save("openaq", f"FIXTURE sensor {sid}", f"openaq/history/sensor_{sid}_p001.json",
                      json.dumps({"sensor": sensor, "request": {}, "response": {"results": results}}).encode("utf-8"))

    fires = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
    for d in range(0, 150, 3):
        day = (START + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        fires += f"31.0,73.9,330,0.4,0.4,{day},0830,N,VIIRS,n,2,290,{rng.uniform(1, 30):.1f},D\n"
    manifest.save("nasa-firms", "FIXTURE fires", "firms/fixture.csv", fires.encode("utf-8"))

    html = """<html><body><main>
      <h2>Unhealthy</h2><p>Everyone may begin to experience health effects. Sensitive groups such as children,
      older adults and people with heart or lung disease should avoid prolonged or heavy exertion outdoors.</p>
      <h2>Good</h2><p>Air quality is satisfactory and air pollution poses little or no risk. Enjoy outdoor activities.</p>
      <h2>Who is at risk</h2><p>Children and teenagers are at greater risk because their lungs are still developing.</p>
      </main></body></html>"""
    manifest.save("docs", "FIXTURE doc", "docs/airnow_aqi_basics.html", html.encode("utf-8"))
    return {"truth": truth, "ts": ts}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A clean SAAFHAWA_HOME with a test config whose split dates fit the fixture."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["train_end"] = "2024-03-31"
    cfg["model"]["valid_end"] = "2024-04-30"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("SAAFHAWA_HOME", str(tmp_path))
    monkeypatch.setenv("SAAFHAWA_CONFIG", str(cfg_path))
    for k in ("OPENAQ_API_KEY", "FIRMS_MAP_KEY", "HF_TOKEN", "HF_DATASET_REPO", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.setenv(k, "")
    _reset_settings()
    yield tmp_path
    _reset_settings()


@pytest.fixture
def built(home):
    """Fixture raw layer, built into the warehouse."""
    from saafhawa import warehouse
    info = write_fixture_raw(home)
    summary = warehouse.build("run-test")
    return {"home": home, "summary": summary, **info}


@pytest.fixture(scope="session")
def trained_home(tmp_path_factory):
    """One trained fixture model shared by the slower tests."""
    from saafhawa import model, warehouse
    from saafhawa.insight_agent import build_docs
    tmp = tmp_path_factory.mktemp("trained")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["train_end"] = "2024-03-31"
    cfg["model"]["valid_end"] = "2024-04-30"
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    old = {k: os.environ.get(k) for k in ("SAAFHAWA_HOME", "SAAFHAWA_CONFIG", "OPENAQ_API_KEY", "FIRMS_MAP_KEY", "HF_TOKEN", "LLM_API_KEY")}
    os.environ.update(SAAFHAWA_HOME=str(tmp), SAAFHAWA_CONFIG=str(tmp / "config.yaml"), OPENAQ_API_KEY="", FIRMS_MAP_KEY="", HF_TOKEN="", LLM_API_KEY="")
    _reset_settings()
    write_fixture_raw(tmp)
    warehouse.build("run-train")
    build_docs()
    # A smaller grid keeps the test fast; the real grid is exercised the same way.
    saved_grid = list(model.GRID)
    model.GRID[:] = [{"learning_rate": 0.1, "max_iter": 120, "max_leaf_nodes": 31, "min_samples_leaf": 40},
                     {"learning_rate": 0.1, "max_iter": 120, "max_leaf_nodes": 15, "min_samples_leaf": 80}]
    result = model.train()
    yield {"home": tmp, "result": result}
    model.GRID[:] = saved_grid
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _reset_settings()
    shutil.rmtree(tmp, ignore_errors=True)

"""Features (no leakage), the split, the baseline and the model, on labelled synthetic fixtures."""

import numpy as np
import pandas as pd


def test_target_is_mean_of_next_24_hours(built):
    from saafhawa import warehouse
    f = warehouse.table("gold_features", "SELECT * FROM gold_features WHERE sensor_id = 102 ORDER BY issued_utc")
    g = warehouse.table("gold_pm25_hourly", "SELECT ts_utc, pm25 FROM gold_pm25_hourly WHERE sensor_id = 102").set_index("ts_utc")["pm25"].sort_index()
    row = f.iloc[500]
    t = row["issued_utc"]
    expected = g[(g.index > t) & (g.index <= t + pd.Timedelta(hours=24))].mean()
    assert np.isclose(row["target_next24"], expected)
    past = g[(g.index > t - pd.Timedelta(hours=24)) & (g.index <= t)].mean()
    assert np.isclose(row["baseline_persistence"], past)  # declared baseline: last 24 h


def test_features_use_only_the_past(built):
    """Changing every PM2.5 value after t must not change any feature at t."""
    from saafhawa import warehouse
    from saafhawa.features import FEATURES, build_features
    pm = warehouse.table("gold_pm25_hourly")
    w, c, fi, st = (warehouse.table(n) for n in ("clean_weather", "clean_cams", "clean_fires", "sensors"))
    t = pm["ts_utc"].min() + pd.Timedelta(days=60)
    a = build_features(pm, w, c, fi, st)
    pm2 = pm.copy()
    pm2.loc[pm2["ts_utc"] > t, "pm25"] *= 3
    b = build_features(pm2, w, c, fi, st)
    cols = [x for x in FEATURES if x != "cams_next24"]  # cams_next24 is a published forecast: documented exception
    ra = a[a.issued_utc == t].set_index("sensor_id")[cols].sort_index()
    rb = b[b.issued_utc == t].set_index("sensor_id")[cols].sort_index()
    assert len(ra) >= 2
    pd.testing.assert_frame_equal(ra, rb)
    # ... while the target does change.
    ta = a[a.issued_utc == t].set_index("sensor_id")["target_next24"].sort_index()
    tb = b[b.issued_utc == t].set_index("sensor_id")["target_next24"].sort_index()
    assert (tb > ta).all()


def test_split_is_time_ordered_with_purge(built):
    from saafhawa import model, warehouse
    df = warehouse.table("gold_features")
    tr, va, te = model.split(df)
    assert tr.issued_utc.max() + pd.Timedelta(hours=24) <= va.issued_utc.min()
    assert va.issued_utc.max() + pd.Timedelta(hours=24) <= te.issued_utc.min()
    assert len(tr) and len(va) and len(te)


def test_model_beats_baseline_with_calibrated_intervals(trained_home):
    r = trained_home["result"]
    t = r["test"]
    assert t["n"] > 500
    assert t["mae_model"] < t["mae_baseline"]
    assert 0.6 <= t["interval_80_coverage"] <= 0.95
    assert "mae_cams" in t and "skill_vs_cams_pct" in t
    assert r["chosen"] in __import__("saafhawa.model", fromlist=["GRID"]).GRID
    assert r["fit_seconds"] > 0 and r["fit_peak_mem_mb"] > 0


def test_errors_by_group_and_reports(trained_home):
    from saafhawa.settings import path
    g = pd.read_csv(path("reports") / "errors_by_group.csv")
    assert set(g["group"]) == {"location_id", "season", "time_of_day", "observed_level", "sensor_type"}
    assert (path("reports") / "metrics.json").exists() and (path("models") / "model.joblib").exists()


def test_forecast_rows_and_ids(trained_home):
    from saafhawa import model, warehouse
    fc = model.forecast("run-fc")
    assert len(fc) == 4 and fc["forecast_id"].is_unique
    assert (fc.p10 <= fc.p50).all() and (fc.p50 <= fc.p90).all()
    assert fc["input_row_id"].str.startswith("feat:").all()
    model.forecast("run-fc-2")  # idempotent: same issue hour replaces, never duplicates
    stored = warehouse.table("gold_forecasts")
    assert stored["forecast_id"].is_unique

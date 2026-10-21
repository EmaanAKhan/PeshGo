"""Train, evaluate and forecast.

Honest evaluation:
- time-based split from config.yaml (train / validation / test), never shuffled
- a 24-hour purge at each boundary, so no training target overlaps a later period
- hyperparameters chosen on validation only; the test set is scored once, at the end
- compared with the declared baseline (persistence) and with CAMS
- uncertainty from quantile models (10th, 50th, 90th percentile), calibrated on
  validation (conformalized quantile regression) so the 80% range really covers ~80%
- errors by station, season, time of day and pollution level
"""

from __future__ import annotations

import json
import time
import tracemalloc
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import __version__, warehouse
from .features import CATEGORICAL, FEATURES
from .settings import config, path

GRID = [
    {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 40},
    {"learning_rate": 0.1, "max_iter": 400, "max_leaf_nodes": 15, "min_samples_leaf": 80},
    {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 63, "min_samples_leaf": 20},
]


def category(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Unknown"
    for c in config()["categories"]:
        if v <= c["max"]:
            return c["name"]
    return config()["categories"][-1]["name"]


def split(df: pd.DataFrame, train_end: str | None = None, valid_end: str | None = None):
    """Time split with a purge. The dates come from config.yaml unless a second experiment passes its own."""
    cfg = config()["model"]
    t_end = pd.Timestamp(train_end or cfg["train_end"], tz="UTC") + pd.Timedelta(days=1)
    v_end = pd.Timestamp(valid_end or cfg["valid_end"], tz="UTC") + pd.Timedelta(days=1)
    purge = pd.Timedelta(hours=cfg["horizon_hours"])
    t = df["issued_utc"]
    train = df[t < t_end - purge]
    valid = df[(t >= t_end) & (t < v_end - purge)]
    test = df[t >= v_end]
    return train, valid, test


def _fit(X, y, q, params, seed):
    cat = [FEATURES.index(c) for c in CATEGORICAL]
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, categorical_features=cat or None, random_state=seed,
                                      early_stopping=False, **params)
    return m.fit(X, y)


def _metrics(y, pred, base, cams, p10=None, p90=None) -> dict:
    out = {"n": int(len(y))}
    if len(y) == 0:
        return out
    mae = float(np.mean(np.abs(pred - y)))
    out["mae_model"] = round(mae, 2)
    out["rmse_model"] = round(float(np.sqrt(np.mean((pred - y) ** 2))), 2)
    ok_b = ~np.isnan(base)
    if ok_b.any():
        mb = float(np.mean(np.abs(base[ok_b] - y[ok_b])))
        out["mae_baseline"] = round(mb, 2)
        out["skill_vs_baseline_pct"] = round(100 * (1 - np.mean(np.abs(pred[ok_b] - y[ok_b])) / mb), 1) if mb else None
    ok_c = ~np.isnan(cams)
    if ok_c.any():
        mc = float(np.mean(np.abs(cams[ok_c] - y[ok_c])))
        out["mae_cams"] = round(mc, 2)
        out["skill_vs_cams_pct"] = round(100 * (1 - np.mean(np.abs(pred[ok_c] - y[ok_c])) / mc), 1) if mc else None
        out["cams_bias"] = round(float(np.mean(cams[ok_c] - y[ok_c])), 2)
    cat_true = np.array([category(v) for v in y])
    out["category_accuracy_model"] = round(float(np.mean(np.array([category(v) for v in pred]) == cat_true)), 3)
    if ok_b.any():
        out["category_accuracy_baseline"] = round(float(np.mean(np.array([category(v) for v in base[ok_b]]) == cat_true[ok_b])), 3)
    if p10 is not None:
        out["interval_80_coverage"] = round(float(np.mean((y >= p10) & (y <= p90))), 3)
        out["interval_80_mean_width"] = round(float(np.mean(p90 - p10)), 1)
    return out


def _groups(df: pd.DataFrame) -> pd.DataFrame:
    """Errors by station, season, day/night and observed level."""
    d = df.copy()
    d["abs_err_model"] = (d["p50"] - d["target_next24"]).abs()
    d["abs_err_baseline"] = (d["baseline_persistence"] - d["target_next24"]).abs()
    d["abs_err_cams"] = (d["cams_next24"] - d["target_next24"]).abs()
    d["season"] = np.where(d["smog_season"] == 1, "smog season (Oct-Jan)", "rest of year")
    d["time_of_day"] = np.where(d["hour_pkt"].between(7, 18), "day (07-18 PKT)", "night")
    d["observed_level"] = d["target_next24"].map(category)
    d["sensor_type"] = np.where(d["is_low_cost"] == 1, "low-cost sensor", "reference monitor")
    rows = []
    for g in ["location_id", "season", "time_of_day", "observed_level", "sensor_type"]:
        for key, part in d.groupby(g, dropna=False):
            rows.append({
                "group": g, "value": str(key), "n": len(part),
                "mae_model": round(part["abs_err_model"].mean(), 2),
                "mae_baseline": round(part["abs_err_baseline"].mean(), 2),
                "mae_cams": round(part["abs_err_cams"].mean(), 2) if part["abs_err_cams"].notna().any() else None,
                "interval_80_coverage": round(float(((part["target_next24"] >= part["p10"]) & (part["target_next24"] <= part["p90"])).mean()), 3),
            })
    return pd.DataFrame(rows)


def train(score_test: bool = True, train_end: str | None = None, valid_end: str | None = None, tag: str = "",
          stride: int = 1, grid: list | None = None) -> dict:
    """Tune on validation, refit, score the test period once. With `tag` (and other dates) it is a separate
    experiment: every output file gets the tag in its name, so the declared-split model is never overwritten."""
    cfg = config()
    seed = cfg["model"]["seed"]
    df = warehouse.table("gold_features")
    df = df[df["target_next24"].notna()].sort_values("issued_utc").reset_index(drop=True)
    if len(df) < 1000:
        raise RuntimeError(f"Only {len(df)} rows with a target: run make data with OpenAQ data first.")
    tr, va, te = split(df, train_end, valid_end)
    if stride > 1:  # neighbouring hours are nearly identical: thin the training rows only (validation and test stay complete)
        tr = tr[tr["issued_utc"].dt.hour % stride == 0]
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError("Train or validation period is empty: check model.train_end / valid_end in config.yaml against the data range.")
    Xtr, ytr = tr[FEATURES].to_numpy(float), tr["target_next24"].to_numpy(float)
    Xva, yva = va[FEATURES].to_numpy(float), va["target_next24"].to_numpy(float)

    t0 = time.perf_counter()
    tracemalloc.start()
    trials, preds_va = [], []
    for params in (grid or GRID):
        p = _fit(Xtr, ytr, 0.5, params, seed).predict(Xva)
        preds_va.append(p)
        trials.append({"params": params, "valid_mae": round(float(np.mean(np.abs(p - yva))), 3)})
    i_best = min(range(len(trials)), key=lambda i: trials[i]["valid_mae"])
    best, valid_pred = trials[i_best]["params"], preds_va[i_best]
    # Conformalized quantile regression: the raw 10th/90th percentile models
    # are often too narrow when conditions shift, so the interval is widened by
    # the margin that makes it cover 80% of the validation period.
    lo_m, hi_m = _fit(Xtr, ytr, 0.1, best, seed), _fit(Xtr, ytr, 0.9, best, seed)
    vlo, vhi = np.minimum(lo_m.predict(Xva), valid_pred), np.maximum(hi_m.predict(Xva), valid_pred)
    raw_cov = float(np.mean((yva >= vlo) & (yva <= vhi)))
    scores = np.maximum(vlo - yva, yva - vhi)
    level = min(1.0, np.ceil(0.8 * (len(yva) + 1)) / len(yva))
    margin = max(0.0, float(np.quantile(scores, level)))
    valid_metrics = _metrics(yva, valid_pred, va["baseline_persistence"].to_numpy(float), va["cams_next24"].to_numpy(float),
                             vlo - margin, vhi + margin)
    valid_metrics["interval_80_coverage_uncalibrated"] = round(raw_cov, 3)
    valid_metrics["interval_margin_ugm3"] = round(margin, 2)

    # Refit on train + validation with the chosen settings, then score the test set once.
    full = pd.concat([tr, va])
    Xf, yf = full[FEATURES].to_numpy(float), full["target_next24"].to_numpy(float)
    models = {q: _fit(Xf, yf, q, best, seed) for q in cfg["model"]["quantiles"]}
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fit_seconds = time.perf_counter() - t0

    result = {
        "model_version": f"{__version__}+{datetime.now(timezone.utc):%Y%m%dT%H%M}",
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "rows": {"train": len(tr), "valid": len(va), "test": len(te)},
        "periods": {
            "train": [str(tr["issued_utc"].min()), str(tr["issued_utc"].max())],
            "valid": [str(va["issued_utc"].min()), str(va["issued_utc"].max())],
            "test": [str(te["issued_utc"].min()), str(te["issued_utc"].max())] if len(te) else None,
        },
        "baseline": cfg["baseline"],
        "tuning": trials, "chosen": best, "interval_margin_ugm3": round(margin, 2),
        "valid": valid_metrics,
        "fit_seconds": round(fit_seconds, 1), "fit_peak_mem_mb": round(peak / 1e6, 1),
    }
    if score_test and len(te):
        Xte = te[FEATURES].to_numpy(float)
        preds = {q: m.predict(Xte) for q, m in models.items()}
        p10 = np.minimum(preds[0.1], preds[0.5]) - margin
        p50 = preds[0.5]
        p90 = np.maximum(preds[0.9], preds[0.5]) + margin
        result["test"] = _metrics(te["target_next24"].to_numpy(float), p50, te["baseline_persistence"].to_numpy(float),
                                  te["cams_next24"].to_numpy(float), p10, p90)
        out = te[["row_id", "location_id", "issued_utc", "target_next24", "baseline_persistence", "cams_next24", "smog_season", "hour_pkt", "is_low_cost"]].copy()
        out["p10"], out["p50"], out["p90"] = p10, p50, p90
        out.to_parquet(path("reports") / f"test_predictions{tag}.parquet", index=False)
        groups = _groups(out)
        groups.to_csv(path("reports") / f"errors_by_group{tag}.csv", index=False)
        worst = groups[groups["n"] >= 50].sort_values("mae_model", ascending=False).head(1)
        result["weakest_group"] = worst.to_dict("records")[0] if len(worst) else None
    joblib.dump({"models": models, "features": FEATURES, "version": result["model_version"], "margin": margin}, path("models") / f"model{tag}.joblib")
    result["split"] = {"train_end": train_end or config()["model"]["train_end"], "valid_end": valid_end or config()["model"]["valid_end"], "tag": tag or "declared", "train_stride_hours": stride}
    (path("reports") / f"metrics{tag}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def rescore() -> dict:
    """Score the saved model on the current test period, without retraining.

    The model was fitted on data up to the validation end only, so every later row is unseen. Rescoring
    is how the test result grows when more monitors' history is added after training; the training and
    validation figures, the chosen settings and the interval margin are kept exactly as they were.
    """
    bundle = joblib.load(path("models") / "model.joblib")
    df = warehouse.table("gold_features")
    df = df[df["target_next24"].notna()].sort_values("issued_utc").reset_index(drop=True)
    _, _, te = split(df)
    if te.empty:
        raise RuntimeError("No rows after the validation period: nothing to score.")
    margin = bundle.get("margin", 0.0)
    X = te[bundle["features"]].to_numpy(float)
    preds = {q: m.predict(X) for q, m in bundle["models"].items()}
    p10 = np.minimum(preds[0.1], preds[0.5]) - margin
    p50 = preds[0.5]
    p90 = np.maximum(preds[0.9], preds[0.5]) + margin
    result = json.loads((path("reports") / "metrics.json").read_text(encoding="utf-8"))
    result["test"] = _metrics(te["target_next24"].to_numpy(float), p50, te["baseline_persistence"].to_numpy(float),
                              te["cams_next24"].to_numpy(float), p10, p90)
    result["rows"]["test"] = len(te)
    result["periods"]["test"] = [str(te["issued_utc"].min()), str(te["issued_utc"].max())]
    out = te[["row_id", "location_id", "issued_utc", "target_next24", "baseline_persistence", "cams_next24", "smog_season", "hour_pkt", "is_low_cost"]].copy()
    out["p10"], out["p50"], out["p90"] = p10, p50, p90
    out.to_parquet(path("reports") / "test_predictions.parquet", index=False)
    groups = _groups(out)
    groups.to_csv(path("reports") / "errors_by_group.csv", index=False)
    worst = groups[groups["n"] >= 50].sort_values("mae_model", ascending=False).head(1)
    result["weakest_group"] = worst.to_dict("records")[0] if len(worst) else None
    result["test_rescored_utc"] = datetime.now(timezone.utc).isoformat()
    result["test_sensors"] = int(te["sensor_id"].nunique())
    (path("reports") / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def forecast(run_id: str) -> pd.DataFrame:
    """Forecast the next 24 hours for every sensor's latest issue hour; writes gold_forecasts."""
    bundle = joblib.load(path("models") / "model.joblib")
    df = warehouse.table("gold_features")
    if df.empty:
        return df
    latest = df.sort_values("issued_utc").groupby("sensor_id").tail(1)
    X = latest[bundle["features"]].to_numpy(float)
    p = {q: m.predict(X) for q, m in bundle["models"].items()}
    margin = bundle.get("margin", 0.0)
    out = pd.DataFrame({
        "forecast_id": [f"fc:{rid.split(':', 1)[1]}" for rid in latest["row_id"]],
        "input_row_id": latest["row_id"].to_numpy(),
        "location_id": latest["location_id"].to_numpy(),
        "sensor_id": latest["sensor_id"].to_numpy(),
        "issued_utc": latest["issued_utc"].to_numpy(),
        "valid_from_utc": (latest["issued_utc"] + pd.Timedelta(hours=1)).to_numpy(),
        "valid_to_utc": (latest["issued_utc"] + pd.Timedelta(hours=24)).to_numpy(),
        "p10": np.clip(np.minimum(p[0.1], p[0.5]) - margin, 0, None).round(1),
        "p50": p[0.5].round(1),
        "p90": (np.maximum(p[0.9], p[0.5]) + margin).round(1),
        "baseline_persistence": latest["baseline_persistence"].round(1).to_numpy(),
        "cams_next24": latest["cams_next24"].round(1).to_numpy(),
        "model_version": bundle["version"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc),
    })
    out["category"] = out["p50"].map(category)
    out["category_p90"] = out["p90"].map(category)
    con = warehouse.connect()
    try:
        con.register("_df", out)
        con.execute("CREATE TABLE IF NOT EXISTS gold_forecasts AS SELECT * FROM _df WHERE false")
        con.execute("DELETE FROM gold_forecasts WHERE forecast_id IN (SELECT forecast_id FROM _df)")
        con.execute("INSERT INTO gold_forecasts SELECT * FROM _df")
        con.unregister("_df")
    finally:
        con.close()
    return out

"""The Data Quality Agent.

It runs validation tools (pandera schemas and the checks below), decides one
action per finding, explains it, and logs it to `quality_log`:

  FIX   a safe, certain correction (duplicates, numeric text, unit labels)
  FILL  a short gap estimated by interpolation, marked `is_filled = 1`
  FLAG  left as it is and warned about; flagged values are not used by the
        model (usable = 0) but stay in the cleaned table for audit

Risky decisions are not taken alone: dropping a whole sensor is only
PROPOSED, and waits in data/approvals.csv until a person approves it on the
dashboard (human in the loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandera.pandas as pa

from .settings import config, path

PM25_UNITS = {"µg/m³", "ug/m3", "μg/m³", "µg/m3", "ug/m³"}


@dataclass
class Decision:
    table: str
    check: str
    rule: str
    action: str  # FIX | FILL | FLAG | PROPOSE | PASS
    rows: int
    reason: str
    detail: str = ""


@dataclass
class QualityAgent:
    run_id: str
    decisions: list[Decision] = field(default_factory=list)
    _written: int = 0

    @property
    def count(self) -> int:
        return len(self.decisions)

    @property
    def q(self) -> dict:
        return config()["quality"]

    def log(self, table, check, rule, action, rows, reason, detail=""):
        self.decisions.append(Decision(table, check, rule, action, int(rows), reason, detail))

    # ── tools ─────────────────────────────────────────────────────────────
    def _schema_failures(self, df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
        try:
            schema.validate(df, lazy=True)
            return pd.DataFrame()
        except pa.errors.SchemaErrors as e:
            return e.failure_cases

    # ── PM2.5 (the target) ────────────────────────────────────────────────
    def check_pm25(self, pm: pd.DataFrame, sensors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = self.q
        cols = ["row_id", "sensor_id", "location_id", "ts_utc", "pm25", "is_filled", "usable", "flags"]
        if pm.empty:
            self.log("pm25", "presence", "at least one sensor with hourly values", "FLAG", 0,
                     "No OpenAQ data in the raw layer: the model cannot be trained or scored.",
                     "Add OPENAQ_API_KEY to .env and run make data.")
            return pd.DataFrame(columns=cols), sensors
        df = pm.copy()

        # FIX: numbers delivered as text.
        as_num = pd.to_numeric(df["value_raw"], errors="coerce")
        text_nums = int((df["value_raw"].map(lambda v: isinstance(v, str)) & as_num.notna()).sum())
        if text_nums:
            self.log("pm25", "type", "value must be numeric", "FIX", text_nums, "Numeric values stored as text were converted to numbers; the conversion is exact.")
        df["pm25"] = as_num

        # FIX: duplicates of the same sensor and hour (re-fetches, overlapping pages).
        before = len(df)
        df = df.sort_values("fetch_time").drop_duplicates(["sensor_id", "ts_utc"], keep="last")
        if before - len(df):
            self.log("pm25", "duplicates", "one value per sensor and hour", "FIX", before - len(df),
                     "Repeated sensor-hour records removed, keeping the latest fetch (later fetches carry provider corrections).")

        df["flags"] = ""

        def flag(mask, name):
            df.loc[mask, "flags"] = df.loc[mask, "flags"] + name + ";"

        # FLAG: wrong units for the whole sensor.
        bad_units = ~df["units"].fillna("").isin(PM25_UNITS)
        if bad_units.any():
            flag(bad_units, "units")
            self.log("pm25", "units", "PM2.5 in µg/m³", "FLAG", bad_units.sum(),
                     "Values not in µg/m³ and with no documented conversion; excluded rather than guessed.",
                     ", ".join(sorted(df.loc[bad_units, "units"].fillna("missing").unique())))

        # FLAG: impossible values (schema check with pandera).
        schema = pa.DataFrameSchema({"pm25": pa.Column(float, pa.Check.in_range(q["pm25_min"], q["pm25_max"]), nullable=True)})
        fails = self._schema_failures(df[["pm25"]].astype(float), schema)
        neg = df["pm25"] < q["pm25_min"]
        high = df["pm25"] > q["pm25_max"]
        if neg.any():
            flag(neg, "negative")
            self.log("pm25", "range", f"pm25 >= {q['pm25_min']}", "FLAG", neg.sum(),
                     "Negative concentrations are physically impossible (sensor offset or error); excluded from modelling, kept for audit.")
        if high.any():
            flag(high, "above_max")
            self.log("pm25", "range", f"pm25 <= {q['pm25_max']}", "FLAG", high.sum(),
                     "Readings above 1000 µg/m³ exceed what Lahore monitors have plausibly recorded; cannot be confirmed, so flagged, not removed.")
        if len(fails) and not (neg.any() or high.any()):
            self.log("pm25", "schema", "pandera schema", "FLAG", len(fails), "Schema failures outside the range rules.", str(fails.head(3).to_dict()))

        # FLAG: a sensor stuck on the same value for many hours.
        df = df.sort_values(["sensor_id", "ts_utc"])
        same = df.groupby("sensor_id")["pm25"].transform(lambda s: s.ne(s.shift()).cumsum())
        run_len = df.groupby(["sensor_id", same])["pm25"].transform("size")
        stuck = (run_len >= q["stuck_hours"]) & df["pm25"].notna() & (df["pm25"] > 0)
        if stuck.any():
            flag(stuck, "stuck")
            self.log("pm25", "stuck", f"same value {q['stuck_hours']}+ hours in a row", "FLAG", stuck.sum(),
                     "A real PM2.5 signal changes hour to hour; long identical runs usually mean a frozen sensor. Not used for training.")

        # FLAG (not fix): low-cost sensors read high in very humid air.
        st = sensors.set_index("sensor_id") if len(sensors) else pd.DataFrame()
        if len(st) and "is_monitor" in st:
            low_cost = df["sensor_id"].map(st["is_monitor"]).fillna(True).eq(False)
            df["is_low_cost"] = low_cost.astype(int)
        else:
            df["is_low_cost"] = 0

        df["usable"] = (df["flags"] == "") & df["pm25"].notna()
        df.loc[~df["usable"], "pm25"] = np.nan

        # FILL short gaps, FLAG long ones, on a complete hourly grid per sensor.
        grids = []
        filled_total, long_gaps, long_rows = 0, 0, 0
        for sid, g in df.groupby("sensor_id"):
            g = g.set_index("ts_utc").sort_index()
            full = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")
            g = g.reindex(full)
            g["sensor_id"] = sid
            g["location_id"] = g["location_id"].ffill().bfill()
            g["is_low_cost"] = g["is_low_cost"].ffill().bfill().fillna(0)
            g["flags"] = g["flags"].fillna("missing;")
            g["usable"] = g["usable"].fillna(False).astype(bool)
            missing = g["pm25"].isna()
            run = (missing != missing.shift()).cumsum()
            run_size = missing.groupby(run).transform("size")
            # Only hours the source never sent are filled; a value the agent
            # flagged (negative, stuck, ...) stays excluded and keeps its flag.
            short = missing & (run_size <= q["fill_max_gap_hours"]) & g["flags"].eq("missing;")
            interp = g["pm25"].interpolate(method="time", limit=q["fill_max_gap_hours"], limit_area="inside")
            fill_mask = short & interp.notna()
            g.loc[fill_mask, "pm25"] = interp[fill_mask]
            g["is_filled"] = fill_mask.astype(int)
            g.loc[fill_mask, "usable"] = True
            g.loc[fill_mask, "flags"] = "filled;"
            filled_total += int(fill_mask.sum())
            longs = missing & (run_size > q["flag_gap_hours"])
            if longs.any():
                long_gaps += int(longs.groupby(run).any().sum())
                long_rows += int(longs.sum())
            grids.append(g.rename_axis("ts_utc").reset_index())
        out = pd.concat(grids, ignore_index=True)
        if filled_total:
            self.log("pm25", "gaps", f"gaps of <= {q['fill_max_gap_hours']} hours", "FILL", filled_total,
                     "Short gaps in a smooth hourly series were interpolated in time and marked is_filled = 1.")
        if long_gaps:
            self.log("pm25", "gaps", f"gaps of > {q['flag_gap_hours']} hours", "FLAG", long_rows,
                     f"{long_gaps} long outages left missing: guessing a day or more of pollution would invent data.")

        out["humid_bias_risk"] = 0  # set by apply_humidity once the weather is known

        # PROPOSE: sensors with too little history. Dropping a sensor is risky, so a person decides.
        out, sensors = self._propose_short_sensors(out, sensors)

        out["row_id"] = "pm25:" + out["sensor_id"].astype(int).astype(str) + ":" + out["ts_utc"].dt.strftime("%Y%m%d%H")
        out["location_id"] = out["location_id"].astype("Int64")
        return out[cols + ["is_low_cost", "humid_bias_risk"]].reset_index(drop=True), sensors

    def apply_humidity(self, pm: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        if pm.empty or weather.empty or "rh_pct" not in weather:
            return pm
        rh = pm["ts_utc"].map(weather.set_index("ts_utc")["rh_pct"])
        risk = (pm["is_low_cost"] == 1) & (rh >= self.q["low_cost_humid_rh"]) & pm["pm25"].notna()
        pm["humid_bias_risk"] = risk.astype(int)
        if risk.any():
            self.log("pm25", "humidity_bias", f"low-cost sensor and RH >= {self.q['low_cost_humid_rh']}%", "FLAG", risk.sum(),
                     "Low-cost optical sensors over-read in very humid air. No validated local correction exists, so values are kept and marked for the model and the dashboard.")
        return pm

    def _propose_short_sensors(self, pm: pd.DataFrame, sensors: pd.DataFrame):
        approved = approved_drops()
        counts = pm[pm["usable"]].groupby("sensor_id").size()
        short = counts[counts < config()["openaq"]["min_hours"]].index.tolist()
        self.proposals = [
            {"sensor_id": int(s), "action": "DROP_SENSOR",
             "reason": f"only {int(counts[s])} usable hours (< {config()['openaq']['min_hours']}); too little history to learn from or score fairly"}
            for s in short if int(s) not in approved
        ]
        if self.proposals:
            self.log("pm25", "history", f"sensor has >= {config()['openaq']['min_hours']} usable hours", "PROPOSE", len(self.proposals),
                     "Dropping a sensor changes coverage, so it waits for human approval on the dashboard; until then the sensor is kept.")
        if approved:
            gone = pm["sensor_id"].isin(approved)
            pm.loc[gone, "usable"] = False
            pm.loc[gone, "flags"] = pm.loc[gone, "flags"] + "dropped_by_approval;"
            self.log("pm25", "history", "approved sensor drops", "FLAG", gone.sum(),
                     f"Sensors {sorted(approved)} excluded after a person approved the drop.")
        return pm, sensors

    # ── weather and CAMS ──────────────────────────────────────────────────
    def check_weather(self, w: pd.DataFrame) -> pd.DataFrame:
        if w.empty:
            self.log("weather", "presence", "hourly weather present", "FLAG", 0, "No Open-Meteo weather in the raw layer.")
            return w
        q = self.q
        w = w.copy()
        lo, hi = q["temp_range_c"]
        rules = {
            "temp_c": (lo, hi, f"{lo} to {hi} °C for Lahore"),
            "rh_pct": (0, 100, "0 to 100 %"),
            "wind_kmh": (0, 200, "0 to 200 km/h"),
            "wind_from_deg": (0, 360, "0 to 360 degrees"),
            "blh_m": (0, 6000, "0 to 6000 m"),
            "precip_mm": (0, 300, "0 to 300 mm per hour"),
        }
        for col, (a, b, txt) in rules.items():
            if col not in w:
                continue
            bad = w[col].notna() & ((w[col] < a) | (w[col] > b))
            if bad.any():
                w.loc[bad, col] = np.nan
                self.log("weather", "range", f"{col} {txt}", "FLAG", bad.sum(), "Physically impossible weather values set to missing; the model treats them as unknown.")
        any_missing = False
        for col in rules:
            if col not in w:
                continue
            miss = w[col].isna()
            if not miss.any():
                continue
            any_missing = True
            span = f"{w.loc[miss, 'ts_utc'].min():%Y-%m-%d} to {w.loc[miss, 'ts_utc'].max():%Y-%m-%d}"
            self.log("weather", "completeness", f"{col} present every hour", "FLAG", int(miss.sum()),
                     f"{col} is missing from the source for {int(miss.sum())} hours ({span}). Left missing, not estimated: "
                     "the model handles unknown values, and filling months of data would invent it.", span)
        if not any_missing:
            self.log("weather", "completeness", "every hour has every variable", "PASS", 0, "All weather hours complete.")
        return w

    def check_cams(self, c: pd.DataFrame) -> pd.DataFrame:
        if c.empty:
            self.log("cams", "presence", "hourly CAMS present", "FLAG", 0, "No CAMS PM2.5 in the raw layer: the CAMS comparison is unavailable.")
            return c
        miss = c["cams_pm25"].isna()
        self.log("cams", "completeness", "CAMS value every hour", "PASS" if not miss.any() else "FLAG", int(miss.sum()),
                 "CAMS hours missing; the model treats them as unknown." if miss.any() else "CAMS complete.")
        return c

    def check_freshness(self, pm, weather, cams, fires):
        now = pd.Timestamp.now(tz="UTC")
        stale_h = self.q["stale_hours"]
        self.freshness = {}
        for name, df, col in [("openaq", pm, "ts_utc"), ("open-meteo-weather", weather, "ts_utc"), ("open-meteo-air", cams, "ts_utc"), ("nasa-firms", fires, "ts_utc")]:
            if df is None or df.empty:
                self.freshness[name] = {"latest_utc": None, "age_hours": None, "stale": True}
                continue
            obs = df
            if name == "openaq":
                obs = df[df["usable"]]
            elif name.startswith("open-meteo"):
                obs = df[df["ts_utc"] <= now]
            latest = obs[col].max() if len(obs) else None
            age = (now - latest).total_seconds() / 3600 if latest is not None else None
            limit = 36 if name == "nasa-firms" else stale_h
            stale = age is None or age > limit
            self.freshness[name] = {"latest_utc": latest.isoformat() if latest is not None else None, "age_hours": round(age, 1) if age is not None else None, "stale": stale}
            if stale:
                self.log(name, "freshness", f"latest observation within {limit} h", "FLAG", 0,
                         f"Source looks stale (last observation {self.freshness[name]['latest_utc']}). Forecasts still run but the dashboard marks the input as late.")

    def check_drift(self, con, feats: pd.DataFrame):
        """Drift monitor: last 7 days of PM2.5 against the same calendar months in training."""
        if feats.empty:
            return
        cfg = config()
        train = feats[feats["issued_utc"] <= pd.Timestamp(cfg["model"]["train_end"], tz="UTC")]
        recent_end = feats["issued_utc"].max()
        recent = feats[feats["issued_utc"] > recent_end - pd.Timedelta(days=7)]
        months = recent["issued_utc"].dt.month.unique()
        ref = train[train["issued_utc"].dt.month.isin(months)]["pm25_now"].dropna()
        cur = recent["pm25_now"].dropna()
        if len(ref) < 200 or len(cur) < 48:
            return
        psi = population_stability(ref.to_numpy(), cur.to_numpy())
        self.drift = {"psi": round(float(psi), 3), "reference_months": sorted(int(m) for m in months)}
        action = "FLAG" if psi >= self.q["drift_psi_flag"] else "PASS"
        self.log("gold_features", "drift", f"PSI(last 7 days vs training, same months) < {self.q['drift_psi_flag']}", action, len(cur),
                 f"Population stability index {psi:.3f}. " + ("Recent PM2.5 no longer looks like the training data for these months: treat forecasts with extra caution and consider retraining." if action == "FLAG" else "Recent data looks like the training data."))

    # ── output ────────────────────────────────────────────────────────────
    def write(self, con):
        con.execute("""CREATE TABLE IF NOT EXISTS quality_log (
            run_id VARCHAR, logged_utc TIMESTAMPTZ, table_name VARCHAR, check_name VARCHAR, rule VARCHAR,
            action VARCHAR, rows_affected BIGINT, reason VARCHAR, detail VARCHAR)""")
        now = datetime.now(timezone.utc)
        for d in self.decisions[self._written:]:
            con.execute("INSERT INTO quality_log VALUES (?,?,?,?,?,?,?,?,?)",
                        [self.run_id, now, d.table, d.check, d.rule, d.action, d.rows, d.reason, d.detail])
        self._written = len(self.decisions)
        # The quality log also goes to a file that survives warehouse rebuilds.
        log_path = path("data") / "quality_log.csv"
        new = not log_path.exists()
        fresh = self.decisions[getattr(self, "_filed", 0):]
        if fresh:
            pd.DataFrame([{"run_id": self.run_id, "logged_utc": now.isoformat(), **d.__dict__} for d in fresh]).to_csv(
                log_path, mode="a", header=new, index=False)
        self._filed = len(self.decisions)
        for p in getattr(self, "proposals", []):
            add_proposal(p["sensor_id"], p["action"], p["reason"])
        self.proposals = []


def population_stability(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


# ── human-in-the-loop approvals (data/approvals.csv, kept across rebuilds) ──

APPROVAL_FIELDS = ["proposal_id", "created_utc", "sensor_id", "action", "reason", "status", "decided_by", "decided_utc"]


def approvals() -> pd.DataFrame:
    p = path("data") / "approvals.csv"
    if not p.exists():
        return pd.DataFrame(columns=APPROVAL_FIELDS)
    return pd.read_csv(p, dtype={"decided_by": str, "decided_utc": str})


def _save_approvals(df: pd.DataFrame) -> None:
    df[APPROVAL_FIELDS].to_csv(path("data") / "approvals.csv", index=False)


def add_proposal(sensor_id: int, action: str, reason: str) -> None:
    df = approvals()
    # Never re-ask about a sensor that is pending or that a person already decided on.
    if ((df["sensor_id"] == sensor_id) & (df["action"] == action)).any():
        return
    now = datetime.now(timezone.utc)
    row = {"proposal_id": f"{action}:{sensor_id}:{now:%Y%m%d%H%M}", "created_utc": now.isoformat(), "sensor_id": sensor_id,
           "action": action, "reason": reason, "status": "pending", "decided_by": "", "decided_utc": ""}
    _save_approvals(pd.concat([df, pd.DataFrame([row])], ignore_index=True))


def approved_drops() -> set[int]:
    df = approvals()
    return {int(s) for s in df[(df["status"] == "approved") & (df["action"] == "DROP_SENSOR")]["sensor_id"]}


def decide(proposal_id: str, approve: bool, who: str) -> bool:
    """Record a person's decision on a pending proposal (used by the dashboard)."""
    df = approvals()
    m = (df["proposal_id"] == proposal_id) & (df["status"] == "pending")
    if not m.any():
        return False
    df.loc[m, "status"] = "approved" if approve else "rejected"
    df.loc[m, "decided_by"] = who
    df.loc[m, "decided_utc"] = datetime.now(timezone.utc).isoformat()
    _save_approvals(df)
    return True

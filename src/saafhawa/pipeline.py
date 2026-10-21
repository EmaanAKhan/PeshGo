"""The hourly pipeline, and the Supervisor that coordinates the agents.

One run: fetch new data from every source -> verify the manifest -> rebuild
cleaned and gold layers (the Data Quality Agent runs inside) -> forecast ->
export the dashboard snapshot. The Supervisor decides what to skip: a source
that fails does not stop the run (its data is marked late on the dashboard),
and no forecast is issued from stale PM2.5 without saying so.

Run it on a schedule with Prefect:
    python -m saafhawa.pipeline serve      # cron "5 * * * *" (5 minutes past every hour)
or once, without Prefect:
    python -m saafhawa run
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from datetime import datetime, timezone

from . import manifest, model, snapshot, warehouse
from .settings import path, secrets

RUN_FIELDS = ["run_id", "started_utc", "finished_utc", "status", "seconds", "new_raw_files", "sources_failed",
              "gold_pm25_rows", "feature_rows", "quality_decisions", "forecasts", "forecast_issued_utc", "note"]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")


def fetch_recent(log: dict) -> int:
    """Fetch the newest data from each source; failures are recorded, not fatal."""
    from .sources import firms, openaq, openmeteo
    s = secrets()
    steps = [("open-meteo", openmeteo.fetch_recent)]
    steps.append(("openaq", openaq.fetch_recent) if s.openaq_key else ("openaq", None))
    steps.append(("nasa-firms", firms.fetch_recent) if s.firms_key else ("nasa-firms", None))
    new = 0
    for name, fn in steps:
        if fn is None:
            log["sources_failed"].append(f"{name}: no API key")
            continue
        try:
            new += fn()
        except Exception as e:  # one source failing must not stop the others
            log["sources_failed"].append(f"{name}: {e}"[:200])
            continue
        if name == "openaq" and openaq.LAST_FAILURES:  # some sensors failed, the others were fetched
            log["sources_failed"].append(f"openaq: {len(openaq.LAST_FAILURES)} sensors failed, e.g. {openaq.LAST_FAILURES[0]}"[:200])
    return new


def supervise(run_id: str, fetch: bool = True) -> dict:
    """One complete run. Returns the run record (also appended to data/runs.csv)."""
    t0 = time.perf_counter()
    rec = {k: "" for k in RUN_FIELDS}
    rec.update(run_id=run_id, started_utc=datetime.now(timezone.utc).isoformat(), status="running", sources_failed=[])
    status = {"run_id": run_id}
    try:
        rec["new_raw_files"] = fetch_recent(rec) if fetch else 0
        manifest.verify()
        summary = warehouse.build(run_id)
        rec.update(gold_pm25_rows=summary["gold_pm25_rows"], feature_rows=summary["feature_rows"], quality_decisions=summary["quality_decisions"])
        status["warehouse"] = summary
        if (path("models") / "model.joblib").exists() and summary["feature_rows"]:
            fc = model.forecast(run_id)
            rec["forecasts"] = len(fc)
            rec["forecast_issued_utc"] = str(fc["issued_utc"].max()) if len(fc) else ""
        else:
            rec["note"] = "no trained model yet (run: python -m saafhawa train)"
        rec["status"] = "success" if not rec["sources_failed"] else "success_with_warnings"
    except Exception as e:
        rec["status"] = "failed"
        rec["note"] = f"{type(e).__name__}: {e}"[:300]
        traceback.print_exc()
    finally:
        rec["finished_utc"] = datetime.now(timezone.utc).isoformat()
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        rec["sources_failed"] = " | ".join(rec["sources_failed"])
        _append_run(rec)
        status.update(last_run=rec, freshness=_freshness(), last_success=_last_success())
        try:
            folder = snapshot.export(status)
            snapshot.push(folder)
        except Exception as e:
            print(f"snapshot failed: {e}", file=sys.stderr)
    return rec


def _append_run(rec: dict) -> None:
    p = path("data") / "runs.csv"
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            w.writeheader()
        w.writerow(rec)


def _last_success() -> dict | None:
    p = path("data") / "runs.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    ok = [r for r in rows if r["status"].startswith("success")]
    return ok[-1] if ok else None


def _freshness() -> dict:
    """Latest observation and last fetch per source."""
    now = datetime.now(timezone.utc)
    out = {}
    fetched = {}
    for r in manifest.read():
        fetched[r["source"]] = max(fetched.get(r["source"], ""), r["fetch_time"])
    queries = {
        "openaq": "SELECT max(ts_utc) FROM gold_pm25_hourly WHERE usable",
        "open-meteo-weather": "SELECT max(ts_utc) FROM clean_weather WHERE ts_utc <= now()",
        "open-meteo-air": "SELECT max(ts_utc) FROM clean_cams WHERE ts_utc <= now()",
        "nasa-firms": "SELECT max(ts_utc) FROM clean_fires",
    }
    limits = {"openaq": 3, "open-meteo-weather": 3, "open-meteo-air": 3, "nasa-firms": 36}
    try:
        con = warehouse.connect(read_only=True)
    except Exception:
        con = None
    for src, q in queries.items():
        latest = None
        if con is not None:
            try:
                latest = con.execute(q).fetchone()[0]
            except Exception:
                latest = None
        age = (now - latest).total_seconds() / 3600 if latest is not None else None
        out[src] = {"last_fetch_utc": fetched.get(src), "latest_observation_utc": latest.isoformat() if latest is not None else None,
                    "age_hours": round(age, 1) if age is not None else None, "late": age is None or age > limits[src], "limit_hours": limits[src]}
    if con is not None:
        con.close()
    return out


# ── Prefect ─────────────────────────────────────────────────────────────────

def prefect_flow():
    from prefect import flow, task

    @task(retries=2, retry_delay_seconds=120, name="supervised hourly run")
    def run_task():
        rec = supervise(new_run_id())
        if rec["status"] == "failed":
            raise RuntimeError(rec["note"])
        return rec

    @flow(name="saaf-hawa-hourly", log_prints=True)
    def hourly():
        rec = run_task()
        print(f"{rec['run_id']}: {rec['status']} in {rec['seconds']} s, forecasts={rec['forecasts']}")
        return rec

    return hourly


def ensure_server(url: str = "http://127.0.0.1:4200/api") -> None:
    """Prefect only schedules runs on a real server (an in-process one cannot), so start
    the local server if none is running, then point this process at it."""
    import os
    import subprocess
    import urllib.request

    if os.environ.get("PREFECT_API_URL"):
        return

    def healthy() -> bool:
        try:
            return urllib.request.urlopen(f"{url}/health", timeout=3).status == 200
        except Exception:
            return False

    if not healthy():
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([sys.executable, "-m", "prefect", "server", "start"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags)
        for _ in range(60):
            time.sleep(2)
            if healthy():
                break
        else:
            raise RuntimeError("The Prefect server did not start. Run `prefect server start` in another window and try again.")
    os.environ["PREFECT_API_URL"] = url
    print(f"Prefect server ready at {url.removesuffix('/api')} (watch the runs there)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        ensure_server()
        prefect_flow().serve(name="hourly", cron="5 * * * *")
    else:
        print(supervise(new_run_id()))

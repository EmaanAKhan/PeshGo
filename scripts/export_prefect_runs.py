"""Export the Prefect flow runs of the judged window to prefect_runs.csv.

    python scripts/export_prefect_runs.py                       # default window below
    python scripts/export_prefect_runs.py --start 2026-10-20T16:00 --end 2026-10-21T16:00

Times are PKT (UTC+5). The CSV comes straight from the Prefect API: one row
per flow run of "saaf-hawa-hourly", with its state and timings. It is joined
with data/runs.csv (the pipeline's own log) by start time, so each Prefect run
also shows what it fetched and forecast.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKT = timezone(timedelta(hours=5))
FIELDS = ["flow_run_id", "flow_run_name", "deployment", "state", "expected_start_pkt", "start_pkt", "end_pkt",
          "duration_s", "pipeline_run_id", "pipeline_status", "forecasts", "sources_failed"]


async def _runs(start, end):
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.filters import FlowFilter, FlowFilterName, FlowRunFilter, FlowRunFilterStartTime
    async with get_client() as c:
        out, offset = [], 0
        while True:
            page = await c.read_flow_runs(
                flow_filter=FlowFilter(name=FlowFilterName(any_=["saaf-hawa-hourly"])),
                flow_run_filter=FlowRunFilter(start_time=FlowRunFilterStartTime(after_=start, before_=end)),
                limit=200, offset=offset)
            out += page
            if len(page) < 200:
                return out
            offset += 200


def _pipeline_log():
    p = ROOT / "data" / "runs.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["_t"] = datetime.fromisoformat(r["started_utc"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-20T16:00")
    ap.add_argument("--end", default="2026-10-21T16:00")
    ap.add_argument("--out", default=str(ROOT / "prefect_runs.csv"))
    a = ap.parse_args(argv)
    start = datetime.fromisoformat(a.start).replace(tzinfo=PKT)
    end = datetime.fromisoformat(a.end).replace(tzinfo=PKT)
    runs = sorted(asyncio.run(_runs(start, end)), key=lambda r: r.start_time or r.expected_start_time)
    log = _pipeline_log()
    rows = []
    for r in runs:
        st = r.start_time
        match = min((x for x in log if st and abs((x["_t"] - st).total_seconds()) < 600), default=None,
                    key=lambda x: abs((x["_t"] - st).total_seconds()))
        rows.append({
            "flow_run_id": str(r.id), "flow_run_name": r.name, "deployment": str(r.deployment_id or ""),
            "state": r.state_name, "expected_start_pkt": r.expected_start_time.astimezone(PKT).isoformat() if r.expected_start_time else "",
            "start_pkt": st.astimezone(PKT).isoformat() if st else "", "end_pkt": r.end_time.astimezone(PKT).isoformat() if r.end_time else "",
            "duration_s": round(r.total_run_time.total_seconds(), 1) if r.total_run_time else "",
            "pipeline_run_id": match["run_id"] if match else "", "pipeline_status": match["status"] if match else "",
            "forecasts": match["forecasts"] if match else "", "sources_failed": match["sources_failed"] if match else "",
        })
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["state"] == "Completed" for r in rows)
    hours = int((end - start).total_seconds() // 3600)
    print(f"{len(rows)} flow runs ({ok} completed) in {hours} scheduled hours -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

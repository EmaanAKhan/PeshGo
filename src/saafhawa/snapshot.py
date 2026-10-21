"""The dashboard snapshot.

The Hugging Face Space cannot read the team's DuckDB file, so every pipeline
run writes a small snapshot (latest forecasts, freshness, quality log, run
history, metrics, documents and the retrieval index) and, if HF_TOKEN and
HF_DATASET_REPO are set, uploads it to a Hugging Face dataset repo that the
Space reads. Nothing in the snapshot is typed by hand: it is all exported
from the gold tables and run logs.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import warehouse
from .settings import path, secrets

FILES = ["status.json", "forecasts.parquet", "quality_log.parquet", "runs.csv", "metrics.json", "errors_by_group.csv",
         "gold_docs.parquet", "rag_index.joblib", "approvals.csv", "recent_pm25.parquet", "local_validation.csv"]


def export(status: dict) -> Path:
    out = path("snapshot")
    con = warehouse.connect(read_only=True)
    try:
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if "gold_forecasts" in tables:
            con.execute("SELECT * FROM gold_forecasts WHERE issued_utc >= (SELECT max(issued_utc) FROM gold_forecasts) - INTERVAL 7 DAY").df() \
               .to_parquet(out / "forecasts.parquet", index=False)
        if "quality_log" in tables:
            con.execute("SELECT * FROM quality_log ORDER BY logged_utc DESC LIMIT 300").df().to_parquet(out / "quality_log.parquet", index=False)
        if "gold_docs" in tables:
            con.execute("SELECT * FROM gold_docs").df().to_parquet(out / "gold_docs.parquet", index=False)
        if "gold_pm25_hourly" in tables:
            con.execute("""SELECT ts_utc, median(pm25) AS pm25_median, count(*) AS sensors FROM gold_pm25_hourly
                           WHERE usable AND ts_utc >= (SELECT max(ts_utc) FROM gold_pm25_hourly) - INTERVAL 14 DAY
                           GROUP BY ts_utc ORDER BY ts_utc""").df().to_parquet(out / "recent_pm25.parquet", index=False)
    finally:
        con.close()
    for src, name in [(path("reports") / "metrics.json", "metrics.json"), (path("reports") / "errors_by_group.csv", "errors_by_group.csv"),
                      (path("models") / "rag_index.joblib", "rag_index.joblib"), (path("data") / "runs.csv", "runs.csv"),
                      (path("data") / "approvals.csv", "approvals.csv"), (path("reports") / "local_validation.csv", "local_validation.csv")]:
        if src.exists():
            shutil.copy2(src, out / name)
    (out / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    try:  # the public page; a rendering problem must never stop a pipeline run
        from .static_site import render
        (out / "index.html").write_text(render(out), encoding="utf-8")
    except Exception as e:
        print(f"static page failed: {e}", file=sys.stderr)
    return out


def push(folder: Path) -> str | None:
    s = secrets()
    # Skip when no repo is configured (or the .env.example placeholder is still there).
    if not s.hf_dataset_repo or s.hf_dataset_repo.startswith("your-hf-user/"):
        return None
    from huggingface_hub import HfApi
    api = HfApi(token=s.hf_token or None)  # None: use the cached `hf auth login` token
    api.create_repo(s.hf_dataset_repo, repo_type="dataset", exist_ok=True, private=False)
    run_id = json.loads((folder / "status.json").read_text())["run_id"]
    info = api.upload_folder(folder_path=str(folder), repo_id=s.hf_dataset_repo, repo_type="dataset", commit_message=f"snapshot {run_id}")
    # The public page lives in a (free) static Space; replace its index.html with the freshly rendered one.
    space = s.hf_space_repo or f"{s.hf_dataset_repo.split('/')[0]}/peshgo"
    page = folder / "index.html"
    if page.exists():
        try:
            api.upload_file(path_or_fileobj=str(page), path_in_repo="index.html", repo_id=space, repo_type="space",
                            commit_message=f"page {run_id}")
        except Exception as e:  # e.g. the Space does not exist yet: the dataset upload above still counts
            print(f"static Space not updated ({space}): {e}", file=sys.stderr)
    return str(info)

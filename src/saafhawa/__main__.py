"""Command line.

python -m saafhawa data       rebuild everything from scratch (make data)
python -m saafhawa train      train and evaluate (writes reports/metrics.json)
python -m saafhawa run        one supervised pipeline run (what Prefect schedules)
python -m saafhawa validate   compare with Punjab EPD daily AQI reports
python -m saafhawa ask "..."  ask the Insight Agent
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

from .settings import path, secrets


def cmd_data(args) -> int:
    from . import manifest, warehouse
    from .insight_agent import build_docs, fetch_docs
    from .pipeline import new_run_id
    from .sources import firms, openaq, openmeteo

    t0 = time.perf_counter()
    if args.fresh and path("data").exists():
        # Start from an empty working folder, keeping only the human decisions.
        keep = path("data") / "approvals.csv"
        saved = keep.read_bytes() if keep.exists() else None
        shutil.rmtree(path("data"))
        if saved:
            path("data")
            keep.write_bytes(saved)
        print("Cleared data/ (kept approvals.csv).")
    for name in ("warehouse",):
        p = path(name)
        if p.exists():
            p.unlink()
    have = {r["source"] for r in manifest.read()}
    s = secrets()
    print("1/5 Fetch or reload raw files")
    if args.offline:
        print("   offline: reusing the raw files listed in the manifest")
    else:
        # History downloads resume: windows already in the manifest are skipped.
        print(f"   Open-Meteo history: {openmeteo.fetch_history()} new files")
        print(f"   Open-Meteo recent:  {openmeteo.fetch_recent()} files")
        if "docs" not in have:
            print(f"   Trusted documents:  {fetch_docs()} files")
        failures = []
        if s.openaq_key:
            print("   OpenAQ history (rate-limited; safe to stop and re-run, it resumes):")
            print(f"   OpenAQ history:     {openaq.fetch_history()} new files")
            failures += [f"OpenAQ history {f}" for f in openaq.LAST_FAILURES]
            print(f"   OpenAQ recent:      {openaq.fetch_recent()} files")
            failures += [f"OpenAQ recent {f}" for f in openaq.LAST_FAILURES]
        else:
            print("   OpenAQ: SKIPPED, no OPENAQ_API_KEY in .env (the model needs this data)")
        if s.firms_key:
            print(f"   NASA FIRMS history: {firms.fetch_history()} new files")
            failures += [f"FIRMS {f}" for f in firms.LAST_FAILURES]
            print(f"   NASA FIRMS recent:  {firms.fetch_recent()} files")
        else:
            print("   NASA FIRMS: SKIPPED, no FIRMS_MAP_KEY in .env (fire features will be empty)")
        if failures:
            print(f"   {len(failures)} windows failed and were skipped (run make data again to retry them):")
            for f in failures[:15]:
                print(f"     - {f}")
    print("2/5 Verify SHA-256 of every raw file against the manifest")
    v = manifest.verify()
    print(f"   {v['files']} files, {v['bytes'] / 1e6:.1f} MB, all match")
    print("3/5 Build cleaned and gold layers (Data Quality Agent runs here)")
    summary = warehouse.build(new_run_id())
    print("4/5 Build document sections and the retrieval index")
    docs = build_docs()
    print(f"   {len(docs)} sections")
    print("5/5 Quality summary")
    for k, val in summary.items():
        print(f"   {k}: {val}")
    log = warehouse.table("quality_log", "SELECT action, check_name, sum(rows_affected) AS rows FROM quality_log GROUP BY ALL ORDER BY action, check_name")
    print(log.to_string(index=False))
    _print_coverage()
    print(f"Done in {time.perf_counter() - t0:.1f} s.")
    return 0


def _print_coverage() -> None:
    """Rows with a target in each period of the declared split, by sensor count."""
    from . import model, warehouse
    df = warehouse.table("gold_features", "SELECT sensor_id, issued_utc, target_next24 FROM gold_features WHERE target_next24 IS NOT NULL")
    if df.empty:
        print("   Split coverage: no rows with a target yet (needs OpenAQ data)")
        return
    tr, va, te = model.split(df)
    print("   Split coverage (rows with a target / sensors):")
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        span = f"{part.issued_utc.min():%Y-%m-%d} .. {part.issued_utc.max():%Y-%m-%d}" if len(part) else "-"
        print(f"     {name:<10} {len(part):>8} rows  {part.sensor_id.nunique():>3} sensors  {span}")
    if not len(tr) or not len(va):
        print("   WARNING: train or validation is empty; training will refuse to run until history covers both periods.")


def cmd_train(args) -> int:
    from . import model
    res = model.rescore() if args.rescore else model.train(score_test=not args.no_test, train_end=args.train_end, valid_end=args.valid_end, tag=args.tag, stride=args.stride,
                                                                                                    grid=[json.loads((path('reports') / 'metrics.json').read_text(encoding='utf-8'))['chosen']] if args.chosen_only else None)
    print(json.dumps({k: res.get(k) for k in ("rows", "periods", "chosen", "valid", "test", "weakest_group", "fit_seconds", "fit_peak_mem_mb", "test_sensors")}, indent=2, default=str))
    return 0


def cmd_run(args) -> int:
    from .pipeline import new_run_id, supervise
    rec = supervise(new_run_id(), fetch=not args.offline)
    print(json.dumps(rec, indent=2, default=str))
    return 0 if rec["status"] != "failed" else 1


def cmd_validate(args) -> int:
    from . import validate_local as v
    if not args.offline:
        print(f"Fetched {v.fetch_reports(args.limit)} new EPD reports")
    df = v.compare()
    print(df.to_string(index=False) if len(df) else "No EPD reports in the raw layer.")
    print(json.dumps(v.summary(df), indent=2))
    return 0


def cmd_ask(args) -> int:
    from .insight_agent import InsightAgent
    print(json.dumps(InsightAgent().answer(" ".join(args.question)), indent=2, ensure_ascii=False, default=str))
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="saafhawa")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--fresh", action="store_true", help="delete data/ first and fetch everything again")
    d.add_argument("--offline", action="store_true", help="no network: rebuild from the raw files in the manifest")
    t = sub.add_parser("train")
    t.add_argument("--no-test", action="store_true", help="tune and validate only; do not touch the test set")
    t.add_argument("--train-end", help="experiment: last training day (YYYY-MM-DD), instead of config.yaml")
    t.add_argument("--valid-end", help="experiment: last validation day, instead of config.yaml")
    t.add_argument("--stride", type=int, default=1, help="experiment: use every Nth hour of the training rows")
    t.add_argument("--chosen-only", action="store_true", help="experiment: reuse the settings chosen by the declared run instead of searching again")
    t.add_argument("--tag", default="", help="experiment name added to every output file, e.g. _multisensor")
    t.add_argument("--rescore", action="store_true", help="do not retrain: score the saved model on the current test period (after more history was added)")
    r = sub.add_parser("run")
    r.add_argument("--offline", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--offline", action="store_true")
    v.add_argument("--limit", type=int, default=30)
    a = sub.add_parser("ask")
    a.add_argument("question", nargs="+")
    args = p.parse_args(argv)
    return {"data": cmd_data, "train": cmd_train, "run": cmd_run, "validate": cmd_validate, "ask": cmd_ask}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the written deliverables from the real results, so no number is typed by hand.

    python scripts/build_docs.py            # writes Summary.html/.pdf, datasetproposal.html/.pdf,
                                            # DATASET_DECLARATION.md and optimization_log.md

Reads reports/metrics.json, reports/errors_by_group.csv, reports/local_validation.csv, data/quality_log.csv,
data/runs.csv, prefect_runs.csv and the warehouse. Run it after `make train` and `make validate`.
"""

from __future__ import annotations

import html
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EDGES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
         r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]


def facts() -> dict:
    import duckdb
    m = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    f: dict = {"m": m, "v": m.get("valid") or {}, "t": m.get("test") or {}}
    g = pd.read_csv(ROOT / "reports" / "errors_by_group.csv") if (ROOT / "reports" / "errors_by_group.csv").exists() else pd.DataFrame()
    f["groups"] = g
    con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
    q = lambda s: con.execute(s).fetchone()[0]
    f.update(pm_rows=q("SELECT count(*) FROM clean_pm25"), pm_usable=q("SELECT count(*) FROM gold_pm25_hourly WHERE usable"),
             sensors=q("SELECT count(DISTINCT sensor_id) FROM gold_pm25_hourly WHERE usable"), weather=q("SELECT count(*) FROM clean_weather WHERE ts_utc <= now()"),
             cams=q("SELECT count(*) FROM clean_cams WHERE ts_utc <= now()"), fires=q("SELECT count(*) FROM clean_fires"),
             first=str(q("SELECT min(ts_utc) FROM gold_pm25_hourly WHERE usable"))[:10], last=str(q("SELECT max(ts_utc) FROM gold_pm25_hourly WHERE usable"))[:10])
    con.close()
    ql = pd.read_csv(ROOT / "data" / "quality_log.csv")
    last = ql[ql["run_id"] == ql["run_id"].iloc[-1]]
    f["quality"] = last
    f["proposals"] = int(last.loc[last["action"] == "PROPOSE", "rows"].sum())
    ap = pd.read_csv(ROOT / "data" / "approvals.csv")
    f["pending"] = int((ap["status"] == "pending").sum())
    runs = pd.read_csv(ROOT / "data" / "runs.csv")
    f["runs"] = runs
    f["run_median_s"] = round(statistics.median(runs["seconds"].dropna()), 0) if len(runs) else None
    pr = ROOT / "prefect_runs.csv"
    f["prefect"] = pd.read_csv(pr) if pr.exists() else pd.DataFrame()
    lv = ROOT / "reports" / "local_validation.csv"
    f["local"] = pd.read_csv(lv) if lv.exists() else pd.DataFrame()
    return f


def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}%"


def spct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:+.{d}f}%"


def qsum(f, action, check=None):
    q = f["quality"]
    q = q[q["action"] == action]
    if check:
        q = q[q["check"] == check]
    return int(q["rows"].sum())


def local_line(f) -> str:
    df = f["local"]
    if df.empty or "difference" not in df:
        return ""
    d = df.dropna(subset=["epd_lahore_aqi", "our_aqi_us_epa"])
    if len(d) < 3:
        return ""
    corr = d["epd_lahore_aqi"].corr(d["our_aqi_us_epa"])
    return (f"Against Punjab EPD's official daily Lahore AQI over {len(d)} matching days: mean absolute difference "
            f"{d['difference'].abs().mean():.0f} AQI points, correlation {corr:.2f}.")


def weakest(f) -> list[str]:
    g = f["groups"]
    out = []
    if g.empty:
        return out
    lvl = g[(g["group"] == "observed_level") & (g["n"] >= 50)].sort_values("mae_model", ascending=False).head(1)
    for _, r in lvl.iterrows():
        cmp = "worse than" if r["mae_model"] > r["mae_baseline"] else "better than"
        out.append(f"{r['value']} hours (n = {int(r['n'])}): MAE {r['mae_model']:.1f}, {cmp} the baseline ({r['mae_baseline']:.1f})")
    s = g[g["group"] == "season"]
    if len(s):
        out.append("; ".join(f"{r['value']}: {r['mae_model']:.1f} vs baseline {r['mae_baseline']:.1f}" for _, r in s.iterrows()))
    return out


CSS = """
@page { size: A4; margin: 10mm 11mm; }
:root { --ink:#1b1f24; --muted:#5b6470; --line:#d9dee4; --accent:#0f6b5c; --bad:#b4372f; }
* { box-sizing: border-box; } body { font: 9.1pt/1.34 "Segoe UI", Arial, sans-serif; color: var(--ink); margin: 0; }
h1 { font-size: 16pt; margin: 0; } h1 span { color: var(--accent); font-weight: 600; } .sub { color: var(--muted); margin: 2px 0 6px; }
h2 { font-size: 10pt; margin: 8px 0 3px; color: var(--accent); text-transform: uppercase; letter-spacing: .04em; }
p { margin: 0 0 3px; } table { border-collapse: collapse; width: 100%; margin: 2px 0 3px; }
th, td { border-bottom: 1px solid var(--line); padding: 2px 5px; text-align: left; font-variant-numeric: tabular-nums; }
th { font-weight: 600; color: var(--muted); font-size: 8.3pt; } td.n, th.n { text-align: right; } tr.us td { font-weight: 700; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 11px; } ul { margin: 0 0 3px; padding-left: 14px; } li { margin: 1px 0; }
.bad { color: var(--bad); font-weight: 600; } .foot { color: var(--muted); font-size: 7.8pt; margin-top: 5px; border-top: 1px solid var(--line); padding-top: 3px; }
"""


def summary_html(f) -> str:
    v, t, m = f["v"], f["t"], f["m"]
    ntest = f["groups"].loc[f["groups"]["group"] == "sensor_type", "n"].sum() if not f["groups"].empty else t.get("n")
    pf = f["prefect"]
    ok = int((pf["state"] == "Completed").sum()) if len(pf) else 0
    window = f"{ok} completed hourly Prefect runs are recorded in prefect_runs.csv" if len(pf) else "see prefect_runs.csv"
    w = weakest(f)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PeshGo summary</title><style>{CSS}</style></head><body>
<h1>PeshGo · <bdi lang="ur">پیشگو</bdi> <span>— 24-hour PM2.5 forecast for Lahore</span></h1>
<p class="sub">HackInsight '26 · Data-to-Decision · results from the real-data run, model trained {html.escape(str(m.get('trained_utc', ''))[:16].replace('T', ' '))} UTC</p>
<h2>Problem and decision</h2>
<p>Smog advisories assume you can stay indoors. Outdoor workers, children and their schools can't, and the US EPA guide lists outdoor workers among the groups most exposed. PeshGo forecasts the mean PM2.5 of the next 24 hours, gives a likely range, and turns it into advice in Urdu and English for <b>outdoor workers, parents and schools, or everyone</b>, with a message to share.</p>
<div class="cols"><div>
<h2>Data (real; no synthetic data)</h2>
<table><tr><th>Source</th><th>Role</th><th class="n">Volume</th></tr>
<tr><td>OpenAQ v3, PM2.5 monitors ≤25 km</td><td>target</td><td class="n">{f['pm_rows']:,} rows, {f['sensors']} sensors</td></tr>
<tr><td>Open-Meteo ERA5 weather</td><td>features</td><td class="n">{f['weather']:,} h</td></tr>
<tr><td>Open-Meteo CAMS (Copernicus)</td><td>feature, 2nd baseline</td><td class="n">{f['cams']:,} h</td></tr>
<tr><td>NASA FIRMS fires (not on the track list)</td><td>upwind fires</td><td class="n">{f['fires']:,} detections</td></tr></table>
<p>{f['first']} to {f['last']}. Raw files are kept unchanged with SHA-256 in <code>manifest.csv</code>; <code>make data</code> rebuilds everything.</p>
<h2>Data Quality Agent (latest run)</h2>
<ul><li><b>FIX</b> {qsum(f, 'FIX'):,} rows (duplicates, numeric text)</li><li><b>FILL</b> {qsum(f, 'FILL'):,} hours in gaps of 2 h or less, marked</li>
<li><b>FLAG</b> {qsum(f, 'FLAG'):,} rows: long outages, stuck sensors, humid-air readings from low-cost sensors, missing boundary-layer height</li>
<li><b>PROPOSE</b> {f['proposals']} sensors to drop, <b>waiting for a person</b> ({f['pending']} pending approvals)</li></ul>
</div><div>
<h2>Model against the declared baseline</h2>
<p>Gradient-boosted quantile model (p10/p50/p90), time split with a 24 h purge, tuned on validation only, <b>test scored once</b>.</p>
<table><tr><th>Test period ({t.get('n', 0):,} rows)</th><th class="n">MAE µg/m³</th><th class="n">Skill vs baseline / CAMS (+ is better)</th></tr>
<tr><td>Persistence (declared baseline)</td><td class="n">{t.get('mae_baseline')}</td><td class="n">—</td></tr>
<tr><td>CAMS forecast</td><td class="n">{t.get('mae_cams')}</td><td class="n">—</td></tr>
<tr class="us"><td>PeshGo</td><td class="n">{t.get('mae_model')}</td><td class="n">{spct(t.get('skill_vs_baseline_pct'))} / {spct(t.get('skill_vs_cams_pct'))}</td></tr></table>
<p>Validation (Nov 2024 to Mar 2025, winter): baseline {v.get('mae_baseline')}, CAMS {v.get('mae_cams')}, PeshGo {v.get('mae_model')} ({spct(v.get('skill_vs_baseline_pct'))} vs baseline). Rows used: training {int(m.get('rows', {}).get('train', 0)):,}, validation {int(m.get('rows', {}).get('valid', 0)):,}, test {int(m.get('rows', {}).get('test', 0)):,}. Category accuracy {t.get('category_accuracy_model', 0) * 100:.1f}% (baseline {t.get('category_accuracy_baseline', 0) * 100:.1f}%).
The 80% range covers {t.get('interval_80_coverage', 0) * 100:.0f}% of test outcomes (aimed at 80%; raw quantiles covered {v.get('interval_80_coverage_uncalibrated', 0) * 100:.0f}% on validation before calibration).</p>
<h2>Errors by group</h2><ul>{''.join(f"<li>{html.escape(x)}</li>" for x in w)}</ul>
</div></div>
<h2>Agents, app and pipeline</h2>
<p>Insight Agent: retrieval over WHO, US EPA AirNow and the Punjab Clean Air Action Plan; forecast numbers come only from stored rows and are shown with their row IDs; every claim is cited; false-premise questions are corrected; off-topic gets "I don't know"; Urdu in and out. Supervisor: one failing source never stops a run. Drift monitor and human approval are built in. The public app is a phone-style page (Urdu or English, share button, installable) that each pipeline run rebuilds; {html.escape(window)}.
Median pipeline run {f['run_median_s']:.0f} s. {html.escape(local_line(f))}</p>
<h2>Notes</h2>
<ul><li>The forecast is the city-wide 24-hour average across Lahore's monitors, not a single street. Readings from low-cost sensors in humid air are marked, not corrected.</li>
<li>Information, not medical advice: follow Punjab government and school notices. The Urdu wording was drafted with AI help and the app says so.</li></ul>
<p class="foot">Training {m.get('fit_seconds')} s, {m.get('fit_peak_mem_mb')} MB peak. AI use is disclosed in AI_Use_Log.md. Code, manifest and logs: github.com/EmaanAKhan/Shaoor.</p></body></html>"""


def proposal_html(f) -> str:
    m = f["m"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PeshGo dataset proposal</title><style>{CSS} body{{font-size:10pt;line-height:1.45}} h2{{margin-top:12px}}</style></head><body>
<h1>PeshGo · <bdi lang="ur">پیشگو</bdi> <span>— dataset proposal</span></h1><p class="sub">The full declaration is DATASET_DECLARATION.md in the repository.</p>
<h2>Question</h2><p>What will the mean PM2.5 in Lahore be over the next 24 hours, how sure are we, and what should an outdoor worker, a parent or a school do about it?</p>
<h2>Why it matters</h2><p>Lahore is regularly among the most polluted cities in the world from October to January. The decision (shift work, hold assembly indoors, send a child with asthma to school) is made the evening before, and advisories assume people can stay indoors. The US EPA guide lists outdoor workers among the groups most exposed.</p>
<h2>Sources (real data only)</h2>
<table><tr><th>Source</th><th>Use</th><th class="n">Volume</th></tr>
<tr><td>OpenAQ v3: PM2.5 monitors within 25 km of Lahore</td><td>target and lags</td><td class="n">{f['pm_rows']:,} hourly rows from {f['sensors']} sensors</td></tr>
<tr><td>Open-Meteo: ERA5 weather archive and forecast</td><td>features</td><td class="n">{f['weather']:,} hourly rows</td></tr>
<tr><td>Open-Meteo: Copernicus CAMS air quality</td><td>feature, second baseline</td><td class="n">{f['cams']:,} hourly rows</td></tr>
<tr><td>NASA FIRMS VIIRS fires (not on the track list)</td><td>upwind fire counts</td><td class="n">{f['fires']:,} detections</td></tr>
<tr><td>Punjab EPD daily AQI reports</td><td>validation only</td><td class="n">daily PDFs</td></tr></table>
<p>Coverage {f['first']} to {f['last']}. More than 50,000 target rows from OpenAQ alone.</p>
<h2>Target, split and baseline (fixed in config.yaml before any result)</h2>
<p>Target: mean PM2.5 of hours t+1 to t+24 per monitor, with at least 18 observed hours. Split by time, never shuffled: train to 31 Oct 2024, validation to 31 Mar 2025, test after, with a 24 h purge at each boundary. Declared baseline: persistence (the next 24 h equal the last 24 h); second reference: the CAMS forecast.</p>
<h2>Reproducibility</h2><p><code>make data</code> fetches every source, saves raw files unchanged with SHA-256 in <code>manifest.csv</code>, and rebuilds the cleaned and gold layers. Settings live in <code>config.yaml</code>; keys only in <code>.env</code>.</p>
<h2>Ethics</h2><p>Public, non-personal data. Advice is information, not medical advice, and defers to Punjab government and school notices. No synthetic data is used for training or evaluation. Low-cost sensors are marked in humid hours, not corrected.</p></body></html>"""


def declaration_md(f) -> str:
    return f"""*Source 1: name, publisher, URL, licence, update frequency*
OpenAQ API v3 (PM2.5 monitors within 25 km of Lahore), OpenAQ Inc, https://docs.openaq.org/ , licence set by each data provider (recorded in every raw file), hourly. Real-time source listed under the Air quality track.

*Source 2: name, publisher, URL, licence, update frequency*
Open-Meteo (ERA5 weather archive and forecast; Copernicus CAMS air quality), Open-Meteo GmbH, https://open-meteo.com/ , CC BY 4.0, hourly, non-commercial free. Real-time source listed under the Climate and heat-health track.

*Which source is not on the track list, and why is it credible?*
NASA FIRMS, VIIRS active-fire detections (Suomi NPP), NASA LANCE / ESDIS, https://firms.modaps.eosdis.nasa.gov/ , NASA open data, near-real-time. It is a primary publisher (the satellite operator), not a re-upload, with documented algorithms and global coverage; I use fire counts and radiative power upwind of Lahore as a crop-burning signal. Punjab EPD's daily AQI reports (https://epd.punjab.gov.pk/air_quality , Government of the Punjab) are used only to validate our PM2.5 locally, never for training.

*Expected volume*
OpenAQ: {f['pm_rows']:,} hourly rows from {f['sensors']} monitors ({f['first']} to {f['last']}). Open-Meteo weather: {f['weather']:,} hourly rows; CAMS: {f['cams']:,} hourly rows. NASA FIRMS: {f['fires']:,} detections. Well above the 50,000-row minimum from OpenAQ alone.

*Why this data fits the problem and the primary user*
Problem: smog advisories in Lahore assume people can stay indoors, but outdoor workers (riders, drivers, construction workers, vendors), children and schools cannot, and the US EPA guide lists outdoor workers among the groups most exposed. Primary user: an outdoor worker, a parent or a school deciding the evening before whether to shift work or move activities indoors. PM2.5 from ground monitors is the quantity the guidance is written in; weather, satellite-model air quality and upwind fires explain where it is heading.

*Baseline you declare (must be fixed before you see results)*
Baseline: persistence, the mean PM2.5 of the next 24 hours equals the mean of the last 24 hours for the same monitor (`baseline: persistence_24h` in config.yaml). A second reference is the CAMS forecast for the same 24 hours. The time split (train to 31 Oct 2024, validation to 31 Mar 2025, test after, 24 h purge) was fixed in config.yaml before any model was trained, and the test set was scored once.

*Change note*
This repository's first declaration (19 Oct 2026, 11:33 PKT) described a different project, a dementia-risk model for Quetta using Pakistan Bureau of Statistics and PMD data. I changed to the project above afterwards; the earlier text is in the git history. Everything on this page describes the project I actually built.
"""


def optimization_md(f) -> str:
    m, v, t = f["m"], f["v"], f["t"]
    trials = "\n".join(f"| {i + 1} | learning rate {x['params']['learning_rate']}, {x['params']['max_iter']} rounds, {x['params']['max_leaf_nodes']} leaves, min {x['params']['min_samples_leaf']} per leaf | {x['valid_mae']} | {'yes' if x['params'] == m.get('chosen') else 'no'} |"
                       for i, x in enumerate(m.get("tuning", [])))
    return f"""# Optimization log

Everything here was measured on the **validation** period; the test set was scored once at the end. The numbers are copied from `reports/metrics.json`.

## Reference points (validation, MAE in µg/m³)
| Predictor | MAE |
|---|---|
| Persistence baseline (declared before results) | {v.get('mae_baseline')} |
| CAMS forecast | {v.get('mae_cams')} |
| **Chosen model** | **{v.get('mae_model')}** |

## 1. Hyper-parameter search (median model, trained on the training period only)
| # | Settings | Validation MAE | Chosen |
|---|---|---|---|
{trials}

## 2. Calibrating the 80% range (conformal adjustment)
The raw 10th and 90th percentile models covered **{v.get('interval_80_coverage_uncalibrated', 0) * 100:.0f}%** of validation outcomes. Widening the range by {v.get('interval_margin_ugm3')} µg/m³, the margin that reaches 80% on validation, gave **{v.get('interval_80_coverage', 0) * 100:.0f}%** on validation and **{t.get('interval_80_coverage', 0) * 100:.0f}%** on the test period (too wide there, because validation is winter only). Kept, because narrow ranges that miss are worse than wide ranges that hold.

## 3. Efficiency
| Item | Value |
|---|---|
| Training time (all fits) | {m.get('fit_seconds')} s on a busy laptop CPU |
| Peak memory during training | {m.get('fit_peak_mem_mb')} MB |
| Median hourly pipeline run | {f['run_median_s']:.0f} s |

"""


def demo_md(f) -> str:
    t, v = f["t"], f["v"]
    w = weakest(f)
    worst = w[0] if w else "the most polluted hours"
    return f"""# Demo video script (target 2 min 40 s, limit 3 min)

Record the phone screen for the app and the laptop for the rest. Say the numbers exactly as written; they come from `reports/metrics.json`.

| Time | Show | Say |
|---|---|---|
| 0:00 to 0:15 | Photo or title card: a rider in Lahore smog (or just the PeshGo title) | "Smog advisories assume you can stay indoors. Delivery riders, drivers, construction workers and children can't. PeshGo tells them what tomorrow's air means for them." |
| 0:15 to 0:55 | Phone, the live app. Switch Workers, then Parents and schools. Tap the Urdu button. Scroll to the hour-by-hour strip. Tap Share on WhatsApp | "One screen. A forecast for the next 24 hours, a likely range, and what to do, in English or Urdu. The advice follows the US EPA guide, which lists outdoor workers among the most exposed. The strip shows the cleaner hours of the last two weeks, and one tap shares it." |
| 0:55 to 1:25 | Laptop: Prefect UI with the hourly runs; `manifest.csv`; the Approvals tab | "Behind it, a pipeline runs every hour. Every raw file is stored unchanged with a SHA-256 hash. A data-quality agent fixes, fills or flags problems and logs why, and risky decisions, like dropping a sensor, wait here for a person to approve." |
| 1:25 to 2:00 | The Evidence section: model against baselines | "On data the model never saw, the last-24-hours baseline we declared up front has an error of {t.get('mae_baseline')}. The satellite forecast, CAMS, has {t.get('mae_cams')}. PeshGo has {t.get('mae_model')}: {pct(t.get('skill_vs_baseline_pct'))} better than the baseline and {pct(t.get('skill_vs_cams_pct'))} better than CAMS. The likely range holds {t.get('interval_80_coverage', 0) * 100:.0f}% of outcomes. The app also shows the errors by group, for example {worst}." |
| 2:00 to 2:25 | Local app, Ask tab: "Is smog good for my health?", then "What is the capital of France?", then an Urdu question | "Ask it something false and it corrects you, with the source. Ask it something outside its documents and it says I don't know. Forecast numbers come only from stored data rows, never from a language model." |
| 2:25 to 2:40 | Title card with the repo and live link | "Limits: the forecast is a city-wide average, the Urdu needs a native reader, and training history is short. Next: more sensors and a field test with real workers. PeshGo, پیشگو." |

## Before recording
- Open the live app on the phone: https://firstladyemaan-peshgo.static.hf.space/
- Run `make dashboard` for the Ask tab and Approvals tab, `make serve` for the Prefect UI at http://127.0.0.1:4200.
"""


def readme_md(f) -> str:
    v, t, m = f["v"], f["t"], f["m"]
    return f"""# PeshGo · پیشگو

**A 24-hour PM2.5 forecast for Lahore, written for the people who can't stay indoors: outdoor workers, children and their schools.**

Live app: https://firstladyemaan-peshgo.static.hf.space/ (a phone-style page that each run of my pipeline rebuilds; it works in English or Urdu and can be added to a phone's home screen).

Smog advisories assume you can stay home. Delivery riders, drivers, construction workers and vendors can't, and the US EPA's own guide lists outdoor workers among the groups most exposed. I forecast the mean PM2.5 of the next 24 hours from ground monitors, give a likely range rather than a single number, and turn it into plain advice in Urdu and English for outdoor workers, for parents and schools, or for everyone, with a message to share. The advice follows the EPA guide row for the forecast category; I make no mask claims because none of my sources covers them.

## Results (from `reports/metrics.json`, built into `Summary.pdf`)
| Test period, Apr 2025 to Oct 2026 ({t.get('n', 0):,} rows) | MAE µg/m³ |
|---|---|
| Persistence baseline (declared before results) | {t.get('mae_baseline')} |
| CAMS satellite-model forecast | {t.get('mae_cams')} |
| **PeshGo** | **{t.get('mae_model')}** ({spct(t.get('skill_vs_baseline_pct'))} vs baseline, {spct(t.get('skill_vs_cams_pct'))} vs CAMS) |

Winter validation (Nov 2024 to Mar 2025): baseline {v.get('mae_baseline')}, CAMS {v.get('mae_cams')}, PeshGo {v.get('mae_model')} ({spct(v.get('skill_vs_baseline_pct'))} vs baseline). The 80% range covers {t.get('interval_80_coverage', 0) * 100:.0f}% of test outcomes. Errors by sensor, season, time of day and pollution level are in `reports/errors_by_group.csv`.

## What is in this repository
| Part | Where | What it does |
|---|---|---|
| Warehouse | `src/saafhawa/warehouse.py`, `manifest.py`, `manifest.csv` | raw files kept exactly as fetched with SHA-256, then cleaned and gold tables with stable row IDs, in DuckDB |
| Data Quality Agent | `quality_agent.py`, `data/quality_log.csv` | fixes, fills or flags each finding and logs why; risky drops wait for a person (`data/approvals.csv`) |
| Model | `features.py`, `model.py` | gradient-boosted quantile model (p10/p50/p90), time split with a 24 h purge, persistence baseline, CAMS comparison, errors by group |
| Insight Agent | `insight_agent.py` | retrieval over WHO, US EPA and Punjab documents; numbers only from stored forecast rows; cited answers; "I don't know" |
| Pipeline | `pipeline.py`, `prefect_runs.csv`, `data/runs.csv` | hourly Prefect flow: fetch, verify, rebuild, forecast, publish; one failing source never stops a run |
| App | `product.py`, `dashboard.py`, `static_site.py` | audience switch, share message, hour-by-hour pattern, evidence section |
| Local validation | `validate_local.py`, `reports/local_validation.csv` | my PM2.5 (as AQI) against Punjab EPD's official daily Lahore AQI |
| Write-up | `Summary.pdf`, `datasetproposal.pdf`, `DATASET_DECLARATION.md`, `optimization_log.md`, `AI_Use_Log.md`, `docs/` | the deliverables |

## Data (real; no synthetic data)
OpenAQ v3 monitors within 25 km of Lahore ({f['pm_rows']:,} hourly rows, {f['sensors']} sensors), Open-Meteo ERA5 weather and Copernicus CAMS air quality, NASA FIRMS fire detections, Punjab EPD daily AQI reports (validation only). The tests use clearly labelled synthetic fixtures (`tests/conftest.py`) and never feed a reported number.

## Run it
```bash
python -m venv .venv && .venv\\Scripts\\activate
pip install -r requirements.txt && pip install -e .
copy .env.example .env            # add OPENAQ_API_KEY and FIRMS_MAP_KEY
make data        # or .\\make.ps1 data: fetch, verify SHA-256, build cleaned and gold layers
make train       # tune on validation, score the test period once
make run         # one supervised run: fetch, rebuild, forecast, publish
make serve       # the hourly Prefect schedule (starts the local Prefect server)
make dashboard   # local app with the question box and the approval buttons
make validate    # compare with Punjab EPD reports
make test
```
`requirements.txt` pins SQLAlchemy 2.0.x because Prefect's scheduler silently creates no runs with 2.1. Keys live only in `.env`, which is not committed. To publish the page, log in to Hugging Face and run `python scripts/deploy_space.py --user <your-name>`.

## Notes
The forecast is the city-wide 24-hour average across the monitors, not a single street. Readings from low-cost sensors in humid air are marked, not corrected. This is information, not medical advice: follow Punjab government and school notices. The Urdu wording was drafted with AI help and needs a native reader. Licence: MIT.
"""


def to_pdf(html_path: Path, pdf_path: Path) -> bool:
    exe = next((e for e in EDGES if Path(e).exists()), None)
    if not exe:
        print("No Edge or Chrome found: open", html_path, "in a browser and print to PDF.")
        return False
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
        proc = subprocess.Popen([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
                          f"--print-to-pdf={pdf_path}", html_path.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            time.sleep(1)
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                time.sleep(1)
                break
        proc.terminate()  # the browser keeps its profile files open until it exits
        time.sleep(1)
    return pdf_path.exists()


def me(text: str) -> str:
    """The write-up is in my voice: 'we' and 'our' become 'I' and 'my'."""
    for a, b in ((r"\bWe\b", "I"), (r"\bwe\b", "I"), (r"\bOur\b", "My"), (r"\bour\b", "my")):
        text = re.sub(a, b, text)
    return text


def main() -> int:
    f = facts()
    (ROOT / "Summary.html").write_text(me(summary_html(f)), encoding="utf-8")
    (ROOT / "datasetproposal.html").write_text(me(proposal_html(f)), encoding="utf-8")
    (ROOT / "DATASET_DECLARATION.md").write_text(me(declaration_md(f)), encoding="utf-8")
    (ROOT / "optimization_log.md").write_text(me(optimization_md(f)), encoding="utf-8")
    (ROOT / "docs" / "DEMO_SCRIPT.md").write_text(me(demo_md(f)), encoding="utf-8")
    (ROOT / "README.md").write_text(me(readme_md(f)), encoding="utf-8")
    for name in ("Summary", "datasetproposal"):
        ok = to_pdf(ROOT / f"{name}.html", ROOT / f"{name}.pdf")
        pages = "?"
        if ok:
            try:
                import pdfplumber
                pages = len(pdfplumber.open(ROOT / f"{name}.pdf").pages)
            except Exception:
                pass
        print(f"{name}.pdf: {'ok' if ok else 'FAILED'}, pages: {pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

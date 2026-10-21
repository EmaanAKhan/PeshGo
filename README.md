
# PeshGo · پیشگو

**A 24-hour PM2.5 forecast for Lahore, written for the people who can't stay indoors: outdoor workers, children and their schools.**

Live app: https://firstladyemaan-peshgo.static.hf.space/ (a phone-style page that each run of our pipeline rebuilds; it works in English or Urdu and can be added to a phone's home screen).

## Team and deliverables
**Team name:** CSV Survivors

| Member | Role |
|---|---|
| Malika Batool | Data Collection |
| Emaan Arif Khan | Planning, Testing |
| Aliza Kanwal | Report, Presentation |

| Deliverable | Link |
|---|---|
| Deployed URL | https://firstladyemaan-peshgo.static.hf.space/ |
| Demo video (3 min) | (https://www.youtube.com/shorts/VdjeU0b30UU) |
| Presentation (PPT) | (https://drive.google.com/drive/folders/1FyEuwfj2kLR_faWQofaehed7FNK29iXE?usp=sharing) |
| One-page summary | `Summary.pdf` |
| Final release | tag `v1.0-final` |

Smog advisories assume you can stay home. Delivery riders, drivers, construction workers and vendors can't, and the US EPA's own guide lists outdoor workers among the groups most exposed. We forecast the mean PM2.5 of the next 24 hours from ground monitors, give a likely range rather than a single number, and turn it into plain advice in Urdu and English for outdoor workers, for parents and schools, or for everyone, with a message to share. The advice follows the EPA guide row for the forecast category; we make no mask claims because none of our sources covers them.

## Results (from `reports/metrics.json`, built into `Summary.pdf`)
| Test period, Apr 2025 to Oct 2026 (398,898 rows) | MAE µg/m³ |
|---|---|
| Persistence baseline (declared before results) | 18.28 |
| CAMS satellite-model forecast | 38.98 |
| **PeshGo** | **19.19** (-3.7% vs baseline, +50.8% vs CAMS) |

Winter validation (Nov 2024 to Mar 2025): baseline 46.05, CAMS 88.4, PeshGo 38.35 (+15.8% vs baseline). The 80% range covers 92% of test outcomes. Errors by sensor, season, time of day and pollution level are in `reports/errors_by_group.csv`.

**Where PeshGo is weak.** Over the full test period it is 3.7% worse than persistence, effectively a tie. Hazardous hours are under-predicted (MAE 51.3 vs 37.1 for persistence, and the range holds only 68% of them), though it is still far ahead of CAMS there (149.8). Most low-cost monitors went online after Oct 2024, so training has only 19,095 rows; we kept the split we declared in advance rather than move it to look better. We also removed a sensor-identity feature that gave a better-looking result, because it changed when sensors were added, then retrained and re-scored on the full period.

## What is in this repository
| Part | Where | What it does |
|---|---|---|
| Warehouse | `src/saafhawa/warehouse.py`, `manifest.py`, `manifest.csv` | raw files kept exactly as fetched with source, query, fetch time, SHA-256 and licence, then cleaned and gold tables with stable row IDs, in DuckDB |
| Data Quality Agent | `quality_agent.py`, `data/quality_log.csv` | fixes, fills or flags each finding and logs why; risky drops wait for a person (`data/approvals.csv`) |
| Model | `features.py`, `model.py` | gradient-boosted quantile model (p10/p50/p90), time split with a 24 h purge, persistence baseline, CAMS comparison, errors by group |
| Insight Agent | `insight_agent.py` | retrieval over WHO, US EPA and Punjab documents; numbers only from stored forecast rows; cited answers; "I don't know" |
| Pipeline | `pipeline.py`, `prefect_runs.csv`, `data/runs.csv` | hourly Prefect flow: fetch, verify, rebuild, forecast, publish; one failing source never stops a run |
| App | `product.py`, `dashboard.py`, `static_site.py` | audience switch, share message, hour-by-hour pattern, evidence section |
| Local validation | `validate_local.py`, `reports/local_validation.csv` | our PM2.5 (as AQI) against Punjab EPD's official daily Lahore AQI |
| Write-up | `Summary.pdf`, `datasetproposal.pdf`, `DATASET_DECLARATION.md`, `optimization_log.md`, `AI_Use_Log.md`, `docs/` | the deliverables |

## Pipeline evidence
`prefect_runs.csv` covers Tue 20 Oct 4 PM to Wed 21 Oct 4 PM (PKT) and has not been edited. Prefect dashboard screenshot: `docs/prefect_dashboard.png`.

## Data (real; no synthetic data)
OpenAQ v3 monitors within 25 km of Lahore (543,283 hourly rows, 78 sensors), Open-Meteo ERA5 weather and Copernicus CAMS air quality, NASA FIRMS fire detections, Punjab EPD daily AQI reports (validation only). The tests use clearly labelled synthetic fixtures (`tests/conftest.py`) and never feed a reported number.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
copy .env.example .env            # add OPENAQ_API_KEY and FIRMS_MAP_KEY
make data        # or .\make.ps1 data: fetch, verify SHA-256, rebuild raw, cleaned and processed layers
make train       # tune on validation, score the test period once
make run         # one supervised run: fetch, rebuild, forecast, publish
make serve       # the hourly Prefect schedule (starts the local Prefect server)
make dashboard   # local app with the question box and the approval buttons
make validate    # compare with Punjab EPD reports
make test
```
`requirements.txt` pins SQLAlchemy 2.0.x because Prefect's scheduler silently creates no runs with 2.1. Keys live only in `.env`, which is not committed. To publish the page, log in to Hugging Face and run `python scripts/deploy_space.py --user <your-name>`.

## Notes
The forecast is the city-wide 24-hour average across the monitors, not a single street. Readings from low-cost sensors in humid air are marked, not corrected. This is information, not medical advice: follow Punjab government and school notices. The question box works only in the local app, not on the public page. The Urdu wording was drafted with AI help and needs a native reader. All material AI use is disclosed in `AI_Use_Log.md`. Licence: MIT.
``

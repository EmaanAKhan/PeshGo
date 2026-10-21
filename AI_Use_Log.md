# AI Use Log

AI use was allowed for this event. I used an AI coding assistant (a large language model) throughout this project. This log says what it did and what I decided.

## What the assistant did

| Date (PKT) | Task | Notes |
|---|---|---|
| 18–19 Oct | Compared project ideas against the hackathon guide and checked which data sources can be reached without a key | I chose the Lahore smog forecast |
| 19 Oct | Wrote the pipeline code: source fetchers, DuckDB warehouse and manifest, Data Quality Agent, features, model, Insight Agent, Prefect flow | The tests use labelled synthetic fixtures only; they never feed a reported number |
| 19 Oct | Tested the Insight Agent on the real documents and fixed bad quotes (two-column PDFs, table rows, questions with false premises) | |
| 20 Oct | Helped me run the first real-data pipeline and fix what broke (request timeouts, one failing sensor stopping a run, history downloads that did not resume, an interval that covered too little) | Every fix has a test |
| 20 Oct | Found that Prefect's scheduler silently created no runs with SQLAlchemy 2.1, and pinned 2.0.x | Recorded in `requirements.txt` |
| 20 Oct | Reframed the product around outdoor workers, children and schools; built the phone-style app, the share message, the static public page and the deployment script | |
| 20 Oct | Renamed the project to PeshGo and moved the Hugging Face repositories | |
| 21 Oct | Found that NASA's fire API only accepts 5-day windows (my config asked for 10), that a sensor-identity feature changed whenever a sensor was added, and that my first test result came from a very small test set. Removed the feature, retrained and re-scored on the full test period | The earlier, more flattering numbers were discarded |
| 21 Oct | Generated the summary, proposal, declaration, optimization log and demo script from the real results files, so no number is typed by hand | `scripts/build_docs.py` |

## What I decided
The project choice (Lahore smog), the audience (outdoor workers first, then children and schools), the name PeshGo, the Hugging Face account and repository for deployment, and that this repository's declaration should describe PeshGo instead of my earlier Quetta plan. I run the code and I own every claim in the write-up.

## What the assistant did not do
- **No AI-generated data.** Every row used for training or evaluation comes from the sources in `DATASET_DECLARATION.md`. Synthetic data exists only in `tests/` as labelled fixtures.
- **No AI-generated numbers in answers.** The Insight Agent reads forecast numbers from `gold_forecasts` rows in code and shows their row IDs. An optional language model may only reword text, and its output is rejected if it contains a number that was not already there (it is off by default).
- **No invented guidance.** Advice follows the US EPA "Air Quality Guide for Particle Pollution". I left out mask advice because none of the documents I cite covers it.
- The split dates and the baseline were fixed in `config.yaml` before any model result was seen.

## Known weakness of AI-written text
The Urdu wording was drafted with AI help and has not been reviewed by a native reader. The app says so.

## Runtime AI
The deployed Insight Agent uses TF-IDF retrieval and quoted sentences, with no language model. `LLM_API_KEY` is optional and off by default.

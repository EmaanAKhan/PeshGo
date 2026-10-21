"""Insight Agent rules, the supervised pipeline and the snapshot (labelled synthetic fixtures)."""

import json
import re

import pandas as pd
import pytest


def test_idk_when_nothing_relevant(trained_home):
    from saafhawa.insight_agent import InsightAgent
    r = InsightAgent().answer("What is the capital of France?")
    assert r["answer_en"].startswith("I don't know") and not r["citations"] and not r["grounded"]


def test_forecast_numbers_come_only_from_rows(trained_home):
    from saafhawa import model, warehouse
    from saafhawa.insight_agent import InsightAgent
    model.forecast("run-ask")
    r = InsightAgent().answer("Should my child go to school tomorrow?")
    fc = warehouse.table("gold_forecasts")
    latest = fc[fc.issued_utc == fc.issued_utc.max()]
    assert set(r["row_ids"]) == set(latest["forecast_id"])
    assert r["numbers_from_rows"]["pm25_p50"] == round(float(latest["p50"].median()))
    assert str(r["numbers_from_rows"]["pm25_p50"]) in r["answer_en"]
    assert r["citations"] and r["grounded"]
    assert r["answer_ur"]  # Urdu output


def test_urdu_question_and_emergency(trained_home):
    from saafhawa.insight_agent import InsightAgent
    r = InsightAgent().answer("کل اسکول؟ بچے کو سانس نہیں آ رہی")
    assert r["emergency"] and "1122" in r["emergency"]["ur"]
    assert r["answer_ur"]


def test_llm_rewording_cannot_add_numbers(trained_home, monkeypatch):
    from saafhawa import insight_agent as ia
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.invalid")
    monkeypatch.setenv("LLM_MODEL", "m")

    class Resp:
        def __init__(self, text):
            self.text = text

        def json(self):
            return {"choices": [{"message": {"content": self.text}}]}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp("PM2.5 will be 999 tomorrow."))
    agent = ia.InsightAgent()
    assert agent._reword("PM2.5 about 80.", [], {}) == "PM2.5 about 80."  # rejected: invented number
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp("Expect PM2.5 near 80."))
    assert agent._reword("PM2.5 about 80.", [], {}) == "Expect PM2.5 near 80."


def test_supervised_run_and_snapshot(trained_home, monkeypatch):
    from saafhawa import pipeline
    from saafhawa.insight_agent import InsightAgent
    from saafhawa.settings import path
    from saafhawa.sources import openmeteo

    def offline():
        raise ConnectionError("network down (simulated)")
    monkeypatch.setattr(openmeteo, "fetch_recent", offline)
    # No keys and no network: every source failure is recorded and the run still completes.
    rec = pipeline.supervise("run-pipe", fetch=True)
    assert rec["status"] == "success_with_warnings", rec["note"]
    assert "openaq: no API key" in rec["sources_failed"] and "network down" in rec["sources_failed"]
    assert rec["forecasts"] == 4
    runs = pd.read_csv(path("data") / "runs.csv")
    assert runs.iloc[-1]["run_id"] == "run-pipe"
    snap = path("snapshot")
    status = json.loads((snap / "status.json").read_text(encoding="utf-8"))
    assert status["last_success"]["run_id"] == "run-pipe"
    assert set(status["freshness"]) == {"openaq", "open-meteo-weather", "open-meteo-air", "nasa-firms"}
    for f in ("forecasts.parquet", "quality_log.parquet", "gold_docs.parquet", "rag_index.joblib", "metrics.json", "runs.csv"):
        assert (snap / f).exists(), f
    # The Space answers from the snapshot alone.
    r = InsightAgent(snapshot_dir=snap).answer("How bad is the air tomorrow?")
    assert r["row_ids"] and r["numbers_from_rows"]


def test_failed_run_is_recorded(trained_home, monkeypatch):
    from saafhawa import pipeline, warehouse
    from saafhawa.settings import path
    monkeypatch.setattr(warehouse, "build", lambda run_id: (_ for _ in ()).throw(ValueError("boom")))
    rec = pipeline.supervise("run-fail", fetch=False)
    assert rec["status"] == "failed" and "boom" in rec["note"]
    runs = pd.read_csv(path("data") / "runs.csv")
    assert runs.iloc[-1]["status"] == "failed"


def test_refetched_document_keeps_unique_section_ids(home):
    from saafhawa import manifest
    from saafhawa.insight_agent import build_docs
    page = "<html><body><main><h2>Unhealthy</h2><p>Sensitive groups such as children, older adults and people with heart or lung disease should avoid prolonged or heavy exertion outdoors today, and everyone else should reduce it.</p></main></body></html>"
    manifest.save("docs", "v1", "docs/airnow_aqi_basics.html", page.encode())
    manifest.save("docs", "v2", "docs/airnow_aqi_basics.html", page.replace("today", "this week").encode())
    docs = build_docs()
    assert docs["chunk_id"].is_unique
    assert docs["text"].str.contains("this week").any() and not docs["text"].str.contains("today").any()

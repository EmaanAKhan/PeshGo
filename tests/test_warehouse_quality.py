"""Raw layer, manifest and the Data Quality Agent (on labelled synthetic fixtures)."""

import pandas as pd
import pytest

from conftest import DUPLICATE, LONG_GAP, NEGATIVE, SHORT_GAP, START, STUCK, TEXT_NUMBER, TOO_HIGH


def _hour(i):
    return START + pd.Timedelta(hours=i)


def test_manifest_records_and_verifies(home):
    from saafhawa import manifest
    a = manifest.save("docs", "q", "docs/x.html", b"hello")
    b = manifest.save("docs", "q", "docs/x.html", b"hello")
    assert a.new and not b.new  # exact duplicate skipped
    c = manifest.save("docs", "q", "docs/x.html", b"changed")
    assert c.new and c.raw_path != a.raw_path  # raw files are immutable: new name
    assert manifest.verify()["files"] == 2
    (home / "data" / a.raw_path).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="1 changed"):
        manifest.verify()


def test_manifest_detects_missing_file(home):
    from saafhawa import manifest
    s = manifest.save("docs", "q", "docs/y.html", b"x")
    (home / "data" / s.raw_path).unlink()
    with pytest.raises(RuntimeError, match="1 missing"):
        manifest.verify()


def _log(built):
    from saafhawa import warehouse
    return warehouse.table("quality_log")


def test_quality_agent_fix_fill_flag_propose(built):
    log = _log(built)
    got = {(r.check_name, r.action) for r in log.itertuples()}
    for expected in [("type", "FIX"), ("duplicates", "FIX"), ("range", "FLAG"), ("stuck", "FLAG"),
                     ("gaps", "FILL"), ("gaps", "FLAG"), ("history", "PROPOSE"), ("humidity_bias", "FLAG")]:
        assert expected in got, expected
    assert log["reason"].str.len().min() > 10  # every decision explains itself


def test_quality_agent_row_level_outcomes(built):
    from saafhawa import warehouse
    g = warehouse.table("gold_pm25_hourly", "SELECT * FROM gold_pm25_hourly WHERE sensor_id = 101").set_index("ts_utc")
    assert g.loc[_hour(TEXT_NUMBER), "usable"]  # text number fixed and kept
    assert not g.loc[_hour(NEGATIVE), "usable"] and "negative" in g.loc[_hour(NEGATIVE), "flags"]
    assert not g.loc[_hour(TOO_HIGH), "usable"] and "above_max" in g.loc[_hour(TOO_HIGH), "flags"]
    assert all("stuck" in g.loc[_hour(i), "flags"] for i in STUCK)
    for i in SHORT_GAP:
        assert g.loc[_hour(i), "is_filled"] == 1 and g.loc[_hour(i), "usable"]
    assert all(not g.loc[_hour(i), "usable"] and g.loc[_hour(i), "is_filled"] == 0 for i in LONG_GAP)
    assert g.index.is_unique  # the duplicate at DUPLICATE was removed
    assert g.loc[_hour(DUPLICATE), "usable"]
    assert g["row_id"].is_unique and g["row_id"].str.startswith("pm25:101:").all()


def test_humidity_flag_only_on_low_cost_sensor(built):
    from saafhawa import warehouse
    g = warehouse.table("gold_pm25_hourly")
    assert g.loc[g.sensor_id == 103, "humid_bias_risk"].sum() > 0
    assert g.loc[g.sensor_id != 103, "humid_bias_risk"].sum() == 0


def test_human_in_the_loop_drop(built):
    from saafhawa import quality_agent as qa, warehouse
    pending = qa.approvals()
    assert list(pending["sensor_id"]) == [104] and (pending["status"] == "pending").all()
    # Rebuilding must not ask again.
    warehouse.build("run-test-2")
    assert len(qa.approvals()) == 1
    # Until approved, the sensor is kept.
    g = warehouse.table("gold_pm25_hourly")
    assert g.loc[g.sensor_id == 104, "usable"].any()
    assert qa.decide(pending["proposal_id"][0], True, "tester")
    assert not qa.decide(pending["proposal_id"][0], True, "tester")  # only pending proposals can be decided
    warehouse.build("run-test-3")
    g = warehouse.table("gold_pm25_hourly")
    assert not g.loc[g.sensor_id == 104, "usable"].any()
    assert g.loc[g.sensor_id == 104, "flags"].str.contains("dropped_by_approval").all()


def test_quality_log_file_survives_rebuild(built):
    from saafhawa import warehouse
    from saafhawa.settings import path
    n1 = len(pd.read_csv(path("data") / "quality_log.csv"))
    warehouse.build("run-test-again")
    n2 = len(pd.read_csv(path("data") / "quality_log.csv"))
    assert n2 > n1


def test_no_openaq_is_flagged_not_crashing(home):
    from conftest import write_fixture_raw
    from saafhawa import manifest, warehouse
    write_fixture_raw(home)
    rows = [r for r in manifest.read() if r["source"] != "openaq"]
    import csv
    with open(home / "data" / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest.FIELDS)
        w.writeheader()
        w.writerows(rows)
    s = warehouse.build("run-no-openaq")
    assert s["gold_pm25_rows"] == 0 and s["feature_rows"] == 0
    log = warehouse.table("quality_log")
    assert ((log.check_name == "presence") & (log.table_name == "pm25")).any()

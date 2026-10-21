"""Fetchers against faked APIs (no network, no keys): timeouts, partial failures, resuming."""

from datetime import datetime, timedelta, timezone

import pytest


class Resp:
    def __init__(self, body):
        self._body = body
        import json
        self.content = json.dumps(body).encode()

    def json(self):
        return self._body


SENSORS = {"results": [
    {"id": 1, "name": "A", "isMonitor": True, "sensors": [{"id": 11, "parameter": {"name": "pm25", "units": "µg/m³"}}],
     "datetimeFirst": {"utc": "2022-01-01T00:00:00Z"}, "datetimeLast": {"utc": datetime.now(timezone.utc).isoformat()}},
    {"id": 2, "name": "B", "isMonitor": False, "sensors": [{"id": 22, "parameter": {"name": "pm25", "units": "µg/m³"}}],
     "datetimeFirst": {"utc": "2022-01-01T00:00:00Z"}, "datetimeLast": {"utc": datetime.now(timezone.utc).isoformat()}},
    {"id": 3, "name": "dead", "isMonitor": True, "sensors": [{"id": 33, "parameter": {"name": "pm25", "units": "µg/m³"}}],
     "datetimeFirst": {"utc": "2019-01-01T00:00:00Z"}, "datetimeLast": {"utc": "2020-01-01T00:00:00Z"}},
]}


@pytest.fixture
def fake_openaq(home, monkeypatch):
    from saafhawa.sources import http, openaq
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    calls = []

    def fake_get(path, params):
        calls.append((path, params))
        if path == "/locations":
            return Resp(SENSORS)
        sid = int(path.split("/")[2])
        a = datetime.fromisoformat(params["datetime_from"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(params["datetime_to"].replace("Z", "+00:00"))
        if sid == 22 and fake_get.fail_22:
            raise http.FetchError(f"{path}: HTTP 500: Internal Server Error")
        if b - a > timedelta(days=fake_get.max_days):
            raise http.FetchError(f"{path}: HTTP 408: Connection timed out")
        return Resp({"results": [{"value": 50.0, "period": {"datetimeFrom": {"utc": params["datetime_from"]}}}]})

    fake_get.fail_22 = False
    fake_get.max_days = 1000
    monkeypatch.setattr(openaq, "_get", fake_get)
    return openaq, calls, fake_get


def test_history_skips_dead_sensors_and_resumes(fake_openaq):
    openaq, calls, _ = fake_openaq
    n1 = openaq.fetch_history()
    assert n1 > 0 and not openaq.LAST_FAILURES
    assert not any("/sensors/33/" in p for p, _ in calls)  # silent since 2020: never asked
    calls.clear()
    n2 = openaq.fetch_history()
    hour_calls = [p for p, _ in calls if p != "/locations"]
    assert n2 <= 2 and len(hour_calls) <= 2  # only the last, unfinished window per sensor is asked again


def test_history_splits_windows_that_time_out(fake_openaq):
    openaq, calls, fake = fake_openaq
    fake.max_days = 30  # the API times out on anything longer than 30 days
    openaq.fetch_history()
    assert not openaq.LAST_FAILURES
    spans = [datetime.fromisoformat(p["datetime_to"][:-1]) - datetime.fromisoformat(p["datetime_from"][:-1])
             for path, p in calls if path != "/locations"]
    assert any(s <= timedelta(days=30) for s in spans)


def test_one_failing_sensor_does_not_stop_the_others(fake_openaq):
    openaq, calls, fake = fake_openaq
    fake.fail_22 = True
    assert openaq.fetch_recent() >= 1  # sensor 11 still fetched
    assert len(openaq.LAST_FAILURES) == 1 and "sensor 22" in openaq.LAST_FAILURES[0]
    openaq.fetch_history()
    assert openaq.LAST_FAILURES and all("sensor 22" in f for f in openaq.LAST_FAILURES)


def test_every_sensor_failing_is_an_error(fake_openaq, monkeypatch):
    openaq, _, fake = fake_openaq
    from saafhawa.sources.http import FetchError
    fake.fail_22 = True
    fake.max_days = -1  # sensor 11 fails too
    with pytest.raises(FetchError, match="every OpenAQ sensor failed"):
        openaq.fetch_recent()


def test_pipeline_records_partial_openaq_failure(fake_openaq, monkeypatch):
    from saafhawa import pipeline
    from saafhawa.sources import openmeteo
    _, _, fake = fake_openaq
    fake.fail_22 = True
    monkeypatch.setattr(openmeteo, "fetch_recent", lambda: 0)
    log = {"sources_failed": []}
    pipeline.fetch_recent(log)
    assert any("1 sensors failed" in f for f in log["sources_failed"])


def test_firms_errors_never_show_the_key(home, monkeypatch):
    from saafhawa.sources import firms, http
    monkeypatch.setenv("FIRMS_MAP_KEY", "SECRETKEY123")

    def boom(url, **k):
        raise http.FetchError(f"{url}: HTTP 500: oops")
    monkeypatch.setattr(firms, "get", boom)
    with pytest.raises(http.FetchError) as e:
        firms.fetch_recent()
    assert "SECRETKEY123" not in str(e.value) and "<MAP_KEY>" in str(e.value)
    firms.fetch_history(until=datetime(2022, 9, 20).date())
    assert firms.LAST_FAILURES and not any("SECRETKEY123" in f for f in firms.LAST_FAILURES)


def test_manifest_has_matches_later_versions(home):
    from saafhawa import manifest
    manifest.save("docs", "q", "x/a.json", b"1")
    manifest.save("docs", "q", "x/a.json", b"2")  # saved as a__<digest>.json
    assert manifest.has("x/a.json") and not manifest.has("x/b.json") and not manifest.has("y/a.json")


def test_firms_history_never_asks_for_more_than_five_days(home, monkeypatch):
    """NASA rejects anything above 5 days per request; a config of 10 once made every history request fail."""
    from datetime import date

    from saafhawa import settings
    from saafhawa.sources import firms
    monkeypatch.setenv("FIRMS_MAP_KEY", "k")
    settings.config()["firms"]["day_range"] = 10
    asked = []

    class R:
        content = b"latitude,longitude,acq_date,acq_time,frp,confidence,satellite\n31,74,2022-09-01,0830,5,n,N\n"

    def fake_get(url, **k):
        asked.append(int(url.split("/")[-2]))
        return R()
    monkeypatch.setattr(firms, "get", fake_get)
    n = firms.fetch_history(until=date(2022, 9, 21))
    assert asked and max(asked) <= 5 and not firms.LAST_FAILURES
    assert n == len(asked) > 3  # parallel fetching still saves every window


def test_two_processes_saving_the_same_file_never_corrupt_the_manifest(home):
    """Separate processes share a file lock, so racing writers cannot leave a checksum that matches no payload."""
    import subprocess
    import sys
    import textwrap

    from saafhawa import manifest
    code = textwrap.dedent("""
        import sys
        from saafhawa import manifest
        for i in range(25):
            manifest.save("docs", "race", "docs/race.json", f'{{"writer": {sys.argv[1]}, "i": {i}}}'.encode())
    """)
    env = dict(__import__("os").environ)
    procs = [subprocess.Popen([sys.executable, "-c", code, str(k)], env=env) for k in range(3)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    assert manifest.verify()["files"] >= 3  # every recorded checksum matches its file

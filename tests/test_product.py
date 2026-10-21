"""The product layer: advice per audience, the hourly pattern, the share text, and the page."""

import pandas as pd
import pytest

from saafhawa import product


def _cats():
    from saafhawa.settings import config
    return [c["name"] for c in config()["categories"]]


def test_every_audience_and_category_has_matching_english_and_urdu(home):
    for a in product.AUDIENCES:
        for c in _cats():
            adv = product.advice(a, c)
            assert adv["headline_en"] and adv["headline_ur"]
            assert adv["bullets_en"] and len(adv["bullets_en"]) == len(adv["bullets_ur"]), (a, c)


def test_workers_get_sensitive_group_guidance_from_unhealthy_up(home):
    """The cited AirNow guide lists outdoor workers as a sensitive group."""
    assert "sensitive group" in product.advice("Outdoor workers", "Unhealthy")["headline_en"].lower()
    vu = " ".join(product.advice("Outdoor workers", "Very unhealthy")["bullets_en"]).lower()
    assert "avoid all physical activity outdoors" in vu
    assert "mask" not in " ".join(sum((product.advice("Outdoor workers", c)["bullets_en"] for c in _cats()), [])).lower()  # no uncited mask advice


def test_text_on_picks_readable_colours():
    assert product.text_on("#d6b100") == "#1b1f24"  # yellow needs dark text
    assert product.text_on("#8e3fb0") == "#ffffff"
    assert product.text_on("#2e9d57") == "#ffffff"


def _recent():
    ts = pd.date_range("2026-10-01", periods=24 * 5, freq="h", tz="UTC")
    pkt_hour = ts.tz_convert("Asia/Karachi").hour
    # clean at 13-15 PKT, worst at 04-06 PKT
    vals = [40 if 13 <= h <= 15 else (220 if 4 <= h <= 6 else 120) for h in pkt_hour]
    return pd.DataFrame({"ts_utc": ts, "pm25_median": vals, "sensors": 5})


def test_hourly_profile_best_and_worst_windows(home):
    prof = product.hourly_profile(_recent())
    assert len(prof) == 24 and prof["days"].max() >= 5
    bw = product.best_and_worst(prof, width=3)
    assert bw["best"] == (13, 16) and bw["worst"] == (4, 7)
    assert product.window_label((13, 16)) == "1 PM to 4 PM"
    assert product.window_label(None) == "n/a"


def test_hourly_profile_handles_no_data(home):
    prof = product.hourly_profile(pd.DataFrame())
    assert prof.empty and "pattern appears once" in product.hours_strip(prof)
    assert product.best_and_worst(pd.DataFrame({"hour": range(24), "pm25": [float("nan")] * 24, "days": 0})) == {"best": None, "worst": None}


def test_share_text_has_both_languages_and_no_mixed_direction_date(home):
    txt = product.share_text("Outdoor workers", 83.0, 135.0, 193.0, "Very unhealthy", "20 Oct, 19:00 PKT", product.hourly_profile(_recent()))
    assert "VERY UNHEALTHY" in txt and "135" in txt and "83" in txt and "193" in txt
    assert "انتہائی مضر صحت" in txt and "Information, not medical advice" in txt
    urdu_line = next(l for l in txt.splitlines() if l.startswith("لاہور کی ہوا"))
    assert "PKT" not in urdu_line and "Oct" not in urdu_line  # left-to-right text inside an Urdu line scrambles it


def _snapshot(tmp_path, with_forecast=True, stale=False):
    import json
    d = tmp_path / "snap"
    d.mkdir()
    if with_forecast:
        now = pd.Timestamp("2026-10-20 12:00", tz="UTC")
        pd.DataFrame({"forecast_id": ["a", "b", "c"], "issued_utc": [now] * 3, "valid_from_utc": [now + pd.Timedelta(hours=1)] * 3,
                      "valid_to_utc": [now + pd.Timedelta(hours=24)] * 3, "p10": [80.0, 90.0, 85.0], "p50": [130.0, 140.0, 135.0],
                      "p90": [190.0, 200.0, 195.0], "model_version": ["t"] * 3}).to_parquet(d / "forecasts.parquet")
    _recent().to_parquet(d / "recent_pm25.parquet")
    fresh = {"openaq": {"late": stale, "age_hours": 30.0 if stale else 1.0, "limit_hours": 3}}
    (d / "status.json").write_text(json.dumps({"freshness": fresh}), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps({"test": {"skill_vs_cams_pct": 53.5, "skill_vs_baseline_pct": 9.8, "interval_80_coverage": 0.95}}), encoding="utf-8")
    return d


def test_page_renders_for_every_audience(home, tmp_path):
    from saafhawa import dashboard
    s = dashboard.Snapshot(_snapshot(tmp_path))
    for a in product.AUDIENCES:
        h = dashboard.product_html(s, a)
        assert "135" in h and "Very unhealthy" in h and "sh-hours" in h and "54% closer" in h
        assert "newest monitor reading is" not in h  # fresh data: no warning
    assert a.lower() in h.lower()
    assert "VERY UNHEALTHY" in dashboard.share_for(s, "Everyone")


def test_page_warns_when_monitor_data_is_stale(home, tmp_path):
    from saafhawa import dashboard
    s = dashboard.Snapshot(_snapshot(tmp_path, stale=True))
    assert "30 hours old" in dashboard.product_html(s, "Outdoor workers")


def test_page_without_a_forecast_says_so(home, tmp_path):
    from saafhawa import dashboard
    s = dashboard.Snapshot(_snapshot(tmp_path, with_forecast=False))
    assert "No forecast yet" in dashboard.product_html(s, "Outdoor workers")
    assert dashboard.share_for(s, "Outdoor workers") == ""


def test_app_builds(home, tmp_path):
    from saafhawa import dashboard
    app = dashboard.build_app(_snapshot(tmp_path), allow_approvals=False)
    assert app is not None


def test_static_app_has_every_audience_both_languages_and_share(home, tmp_path):
    from saafhawa import static_site
    page = static_site.render(_snapshot(tmp_path))
    for a in product.AUDIENCES:
        assert a.replace("&", "&amp;") in page
    assert page.count("data-aud-sec") >= 3 and 'id="share-workers"' in page and "sh-hours" in page
    assert "<script src" not in page  # all the logic is inline: one file
    assert 'data-l="en"' in page and 'data-l="ur"' in page and "data-lang-btn" in page  # one language shown at a time
    assert "Share on WhatsApp" in page and "wa.me" in page
    assert "The evidence" in page and "model against the baselines" in page
    assert "135" in page and "VERY UNHEALTHY" in page and "Very unhealthy" in page
    assert 'rel="manifest"' in page and "serviceWorker" in page  # installable


def test_app_install_files(tmp_path):
    import json

    from saafhawa import static_site
    names = static_site.write_assets(tmp_path)
    assert {"manifest.json", "sw.js"} <= set(names)
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["display"] == "standalone" and m["short_name"] == "PeshGo" and m["start_url"] == "./"
    assert all((tmp_path / i["src"]).exists() for i in m["icons"]) or "icon-192.png" not in names  # icons exist when Pillow does
    assert "fetch" in (tmp_path / "sw.js").read_text(encoding="utf-8")


def test_snapshot_export_writes_the_page_and_push_skips_a_placeholder_repo(home, monkeypatch):
    from saafhawa import snapshot
    monkeypatch.setenv("HF_DATASET_REPO", "your-hf-user/peshgo-snapshot")
    assert snapshot.push(home) is None  # the .env.example placeholder must never be uploaded to

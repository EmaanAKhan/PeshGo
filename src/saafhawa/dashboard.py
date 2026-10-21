"""The PeshGo web app (Gradio).

It reads one snapshot folder, the same files the pipeline exports after every
run, so the local app and the Hugging Face Space show exactly the same thing:

    python -m saafhawa.dashboard            # local, reads data/snapshot, approvals enabled
    space/app.py                            # the Space, reads the snapshot from the HF dataset repo

The first screen is the product: what the next 24 hours of air mean for the
person looking at it (outdoor workers, parents and schools, or everyone), in
English and Urdu, with a message to share. The evidence (model against
baselines, errors by group, data quality, pipeline status, approvals) sits in
the tabs below it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from . import product
from .settings import config

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
:root{--sh-ink:#16202a;--sh-muted:#566170;--sh-surface:#ffffff;--sh-line:#dde3ea;--sh-soft:#f3f6f9;--sh-accent:#0f6b5c;--sh-warn:#9a3412;--sh-warnbg:#fff1e6}
.dark{--sh-ink:#e8edf2;--sh-muted:#a4afbc;--sh-surface:#1d242c;--sh-line:#34404c;--sh-soft:#161c22;--sh-accent:#4cc3ae;--sh-warn:#ffb98a;--sh-warnbg:#3a2616}
.sh-top{display:flex;align-items:center;gap:14px;margin:4px 0 10px}
.sh-logo{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,#0f6b5c,#2e9d57);color:#fff;display:grid;place-items:center;font-weight:800;font-size:18px;flex:none}
.sh-top h1{margin:0;font-size:26px;letter-spacing:-.01em;color:var(--sh-ink)}
.sh-top p{margin:2px 0 0;color:var(--sh-muted);font-size:14px}
.sh-hero{display:grid;grid-template-columns:minmax(260px,5fr) 7fr;border-radius:20px;overflow:hidden;border:1px solid var(--sh-line);background:var(--sh-surface);box-shadow:0 6px 24px rgba(20,30,40,.08)}
@media(max-width:760px){.sh-hero{grid-template-columns:1fr}}
.sh-main{background:var(--c);color:var(--t);padding:22px 24px}
.sh-main *{color:var(--t)}
.sh-eyebrow{font-size:13px;font-weight:600;opacity:.9;text-transform:uppercase;letter-spacing:.06em}
.sh-num{font-size:72px;font-weight:800;line-height:1;margin:8px 0 2px;letter-spacing:-.03em}
.sh-num span{font-size:18px;font-weight:600;opacity:.85;letter-spacing:0;margin-left:6px}
.sh-cat{font-size:24px;font-weight:700;margin-bottom:12px}
.sh-bar{position:relative;height:12px;border-radius:99px;background:rgba(255,255,255,.35);margin:6px 0 2px}
.sh-tick{position:absolute;top:-2px;bottom:-2px;width:1px;background:rgba(0,0,0,.25)}
.sh-range{position:absolute;top:0;bottom:0;border-radius:99px;background:rgba(0,0,0,.45)}
.sh-mid{position:absolute;top:-4px;width:6px;height:20px;border-radius:3px;background:#fff;margin-left:-3px;box-shadow:0 0 0 2px rgba(0,0,0,.45)}
.sh-bar-scale{display:flex;justify-content:space-between;font-size:11px;opacity:.85}
.sh-meta{font-size:13px;margin-top:10px;opacity:.95;line-height:1.5}
.sh-act{padding:22px 26px;color:var(--sh-ink)}
.sh-aud{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--sh-accent);background:var(--sh-soft);border-radius:99px;padding:3px 12px}
.sh-act h2{font-size:25px;line-height:1.25;margin:10px 0 8px;color:var(--sh-ink)}
.sh-act ul{margin:0 0 10px;padding-left:20px;font-size:16px;line-height:1.55;color:var(--sh-ink)}
.sh-ur{direction:rtl;text-align:right;font-family:'Noto Nastaliq Urdu','Jameel Noori Nastaleeq','Segoe UI',serif;font-size:19px;line-height:2.1;background:var(--sh-soft);border-radius:14px;padding:10px 16px;color:var(--sh-ink)}
.sh-ur b{font-size:21px}
.sh-ur ul{padding-right:22px;padding-left:0;margin:0;font-size:18px}
.sh-alert{margin:12px 0 0;border-radius:12px;padding:10px 14px;background:var(--sh-warnbg);color:var(--sh-warn);font-size:14px;font-weight:600}
.sh-sec{margin:18px 0 6px;font-size:13px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--sh-muted)}
.sh-hours{display:grid;grid-template-columns:repeat(24,1fr);gap:3px;align-items:end;background:var(--sh-surface);border:1px solid var(--sh-line);border-radius:14px;padding:12px 12px 6px}
.sh-hr{display:flex;flex-direction:column;align-items:center;gap:3px}
.sh-hr-b{display:block;width:100%;height:34px;border-radius:6px}
.sh-hr small{font-size:10px;color:var(--sh-muted);height:12px}
.sh-best .sh-hr-b{outline:3px solid #14532d;outline-offset:1px}
.sh-worst .sh-hr-b{outline:3px solid #7f1d1d;outline-offset:1px}
.sh-note{font-size:13px;color:var(--sh-muted);margin:6px 2px 0;line-height:1.5}
.sh-trust{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.sh-tile{background:var(--sh-surface);border:1px solid var(--sh-line);border-radius:14px;padding:12px 14px}
.sh-tile b{display:block;font-size:24px;color:var(--sh-accent);letter-spacing:-.01em}
.sh-tile span{font-size:13px;color:var(--sh-muted);line-height:1.4}
.sh-empty{background:var(--sh-soft);border-radius:18px;padding:30px;text-align:center;color:var(--sh-ink)}
.sh-pill{display:inline-block;border-radius:99px;padding:2px 10px;font-size:13px;font-weight:700;margin-right:6px}
.sh-ok{background:#d8f3e1 !important;color:#14532d !important}.sh-late{background:#fde2e1 !important;color:#7f1d1d !important}
.sh-foot{font-size:12.5px;color:var(--sh-muted);margin:16px 2px 4px;line-height:1.55}
"""


# ── reading the snapshot ────────────────────────────────────────────────────

class Snapshot:
    def __init__(self, folder):
        self.dir = Path(folder)

    def _read(self, name):
        p = self.dir / name
        if not p.exists():
            return None
        if name.endswith(".parquet"):
            return pd.read_parquet(p)
        if name.endswith(".csv"):
            try:
                return pd.read_csv(p)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        return json.loads(p.read_text(encoding="utf-8"))

    @property
    def status(self) -> dict:
        return self._read("status.json") or {}

    @property
    def forecasts(self) -> pd.DataFrame:
        fc = self._read("forecasts.parquet")
        if fc is None or fc.empty:
            return pd.DataFrame()
        return fc[fc["issued_utc"] == fc["issued_utc"].max()]

    def table(self, name) -> pd.DataFrame:
        df = self._read(name)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    @property
    def metrics(self) -> dict:
        return self._read("metrics.json") or {}


def _pkt(ts, fmt="%a %d %b %Y, %H:%M PKT") -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "unknown"
    return pd.Timestamp(ts).tz_convert(config()["display_timezone"]).strftime(fmt)


def forecast_numbers(s: Snapshot) -> dict | None:
    """The headline numbers, taken from the latest forecast rows."""
    fc = s.forecasts
    if fc.empty:
        return None
    p50, p10, p90 = (float(fc[c].median()) for c in ("p50", "p10", "p90"))
    return {"p50": p50, "p10": p10, "p90": p90, "n": len(fc), "category": product.category_for(p50),
            "start": fc["valid_from_utc"].min(), "end": fc["valid_to_utc"].max(), "issued": fc["issued_utc"].max(),
            "version": str(fc["model_version"].iloc[0]), "ids": fc["forecast_id"].astype(str).tolist()}


# ── the product screen ──────────────────────────────────────────────────────

def _stale_banner(s: Snapshot) -> str:
    f = (s.status.get("freshness") or {}).get("openaq") or {}
    if f.get("late") and f.get("age_hours") is not None:
        return f"<div class='sh-alert'>The newest monitor reading is {f['age_hours']:.0f} hours old. Treat this forecast with extra care.</div>"
    return ""


def trust_tiles(s: Snapshot) -> str:
    t = s.metrics.get("test") or {}
    tiles = []
    if t.get("skill_vs_cams_pct") is not None:
        tiles.append((f"{t['skill_vs_cams_pct']:.0f}% closer", "than the satellite-model forecast (CAMS) on data the model never saw"))
    if t.get("skill_vs_baseline_pct") is not None:
        tiles.append((f"{t['skill_vs_baseline_pct']:.0f}% closer", "than simply repeating the last 24 hours (our declared baseline)"))
    if t.get("interval_80_coverage") is not None:
        tiles.append((f"{t['interval_80_coverage'] * 100:.0f}% inside", "of real outcomes fell inside the likely range (aimed at 80%)"))
    f = (s.status.get("freshness") or {}).get("openaq") or {}
    if f.get("age_hours") is not None:
        tiles.append((f"{f['age_hours']:.0f} h old", "newest monitor reading; the pipeline re-checks every hour"))
    if not tiles:
        return ""
    body = "".join(f"<div class='sh-tile'><b>{product.esc(a)}</b><span>{product.esc(b)}</span></div>" for a, b in tiles)
    return f"<div class='sh-sec'>Why you can check this</div><div class='sh-trust'>{body}</div>"


def product_html(s: Snapshot, audience: str) -> str:
    n = forecast_numbers(s)
    if n is None:
        return ("<div class='sh-empty'><h2>No forecast yet</h2><p>The pipeline has not produced a forecast. "
                "It needs recent monitor data and a trained model.</p></div>")
    cat = n["category"]
    a = product.advice(audience, cat["name"])
    bullets = "".join(f"<li>{product.esc(b)}</li>" for b in a["bullets_en"])
    bullets_ur = "".join(f"<li>{product.esc(b)}</li>" for b in a["bullets_ur"])
    worse = product.category_for(n["p90"])["name"]
    worse_note = (f"<div class='sh-alert'>The high end of the range is “{product.esc(worse)}”. Plan for that if the work cannot move.</div>"
                  if worse != cat["name"] else "")
    profile = product.hourly_profile(s.table("recent_pm25.parquet"))
    return f"""
    <div class='sh-hero' style='--c:{cat["color"]};--t:{product.text_on(cat["color"])}'>
      <div class='sh-main'>
        <div class='sh-eyebrow'>Next 24 hours · Lahore</div>
        <div class='sh-num'>{n['p50']:.0f}<span>µg/m³ PM2.5</span></div>
        <div class='sh-cat'>{product.esc(cat['name'])} · {product.esc(cat.get('name_ur', ''))}</div>
        {product.range_bar(n['p10'], n['p50'], n['p90'], cat['color'])}
        <div class='sh-meta'>Likely range {n['p10']:.0f} to {n['p90']:.0f} · median of {n['n']} monitors<br>
        {_pkt(n['start'], '%a %H:%M')} to {_pkt(n['end'], '%a %H:%M')} PKT · issued {_pkt(n['issued'], '%H:%M')}</div>
      </div>
      <div class='sh-act'>
        <span class='sh-aud'>For {product.esc(audience.lower())}</span>
        <h2>{product.esc(a['headline_en'])}</h2>
        <ul>{bullets}</ul>
        <div class='sh-ur' lang='ur'><b>{product.esc(a['headline_ur'])}</b><ul>{bullets_ur}</ul></div>
        {worse_note}{_stale_banner(s)}
      </div>
    </div>
    <div class='sh-sec'>The day, hour by hour (what the last two weeks looked like)</div>
    {product.hours_strip(profile)}
    {trust_tiles(s)}
    <p class='sh-foot'>Advice follows the US EPA “Air Quality Guide for Particle Pollution”; outdoor workers are among the people it lists as most exposed.
    The number is the 24-hour average across Lahore's monitors, not your street. Information, not medical advice: follow Punjab government and school notices.
    The Urdu was drafted with AI help; please have a native reader check it before relying on it.</p>"""


def share_for(s: Snapshot, audience: str) -> str:
    n = forecast_numbers(s)
    if n is None:
        return ""
    when = _pkt(n["issued"], "%d %b, %H:%M PKT")
    return product.share_text(audience, n["p10"], n["p50"], n["p90"], n["category"]["name"], when,
                              product.hourly_profile(s.table("recent_pm25.parquet")))


# ── evidence and pipeline panels ────────────────────────────────────────────

def status_panel(s: Snapshot) -> str:
    st = s.status
    last = st.get("last_success") or {}
    run = st.get("last_run") or {}
    rows = []
    for src, f in (st.get("freshness") or {}).items():
        pill = "<span class='sh-pill sh-late'>LATE</span>" if f.get("late") else "<span class='sh-pill sh-ok'>FRESH</span>"
        age = f"{f['age_hours']} h old" if f.get("age_hours") is not None else "no data"
        rows.append(f"<tr><td>{pill}</td><td><b>{html.escape(src)}</b></td><td>{age} (limit {f.get('limit_hours')} h)</td>"
                    f"<td>latest {html.escape(str(f.get('latest_observation_utc') or '—'))}</td></tr>")
    ql = s.table("quality_log.parquet")
    if len(ql) and "run_id" in ql:
        cur = ql[ql["run_id"] == ql.sort_values("logged_utc")["run_id"].iloc[-1]]
        q = " · ".join(f"{k} {v}" for k, v in sorted(cur.groupby("action").size().to_dict().items()))
    else:
        q = "no quality log yet"
    ap = s.table("approvals.csv")
    pending = int((ap["status"] == "pending").sum()) if len(ap) and "status" in ap else 0
    run_pill = "sh-ok" if str(run.get("status", "")).startswith("success") else "sh-late"
    return f"""
    <div>
      <p><b>Last successful run:</b> {html.escape(str(last.get('run_id', 'none yet')))} · finished {html.escape(str(last.get('finished_utc', '—')))}</p>
      <p><b>Latest run:</b> <span class='sh-pill {run_pill}'>{html.escape(str(run.get('status', 'unknown')).upper())}</span>
         {html.escape(str(run.get('run_id', '')))} · {html.escape(str(run.get('seconds', '')))} s ·
         failed sources: {html.escape(str(run.get('sources_failed') or 'none'))}</p>
      <p><b>Quality status (latest run):</b> {html.escape(q)} · <b>pending approvals:</b> {pending}</p>
      <table>{''.join(rows)}</table>
    </div>"""


def metrics_table(s: Snapshot) -> pd.DataFrame:
    m = s.metrics
    rows = []
    for split in ("valid", "test"):
        d = m.get(split) or {}
        if not d.get("n"):
            continue
        rows.append({"period": split, "rows": d["n"], "MAE model": d.get("mae_model"), "MAE baseline (persistence)": d.get("mae_baseline"),
                     "MAE CAMS": d.get("mae_cams"), "skill vs baseline %": d.get("skill_vs_baseline_pct"),
                     "skill vs CAMS %": d.get("skill_vs_cams_pct"), "category accuracy": d.get("category_accuracy_model"),
                     "80% interval coverage": d.get("interval_80_coverage")})
    return pd.DataFrame(rows)


def recent_series(s: Snapshot) -> pd.DataFrame:
    pm = s.table("recent_pm25.parquet")
    if pm.empty:
        return pd.DataFrame({"time (PKT)": [], "PM2.5 µg/m³": [], "series": []})
    out = pd.DataFrame({"time (PKT)": pm["ts_utc"].dt.tz_convert(config()["display_timezone"]).dt.tz_localize(None),
                        "PM2.5 µg/m³": pm["pm25_median"].round(1), "series": "observed (median of monitors)"})
    fc = s.forecasts
    if len(fc):
        t0 = fc["valid_from_utc"].min().tz_convert(config()["display_timezone"]).tz_localize(None)
        t1 = fc["valid_to_utc"].max().tz_convert(config()["display_timezone"]).tz_localize(None)
        for col, name in (("p50", "forecast (24 h mean)"), ("p10", "forecast low (p10)"), ("p90", "forecast high (p90)")):
            v = round(float(fc[col].median()), 1)
            out = pd.concat([out, pd.DataFrame({"time (PKT)": [t0, t1], "PM2.5 µg/m³": [v, v], "series": name})])
    return out


def ask(s: Snapshot, question: str):
    from .insight_agent import InsightAgent
    r = InsightAgent(snapshot_dir=s.dir).answer(question)
    cites = "\n".join(f"- [{html.escape(c['title'])} — {html.escape(str(c['section']))}]({c['url']}) · `{c['chunk_id']}` ({c['for']})"
                      for c in r["citations"]) or "_none_"
    rows = ", ".join(f"`{x}`" for x in r["row_ids"][:6]) or "_none_"
    extra = ""
    if r.get("emergency"):
        extra = f"\n\n> **{r['emergency']['en']}**\n>\n> {r['emergency']['ur']}"
    en = f"{r['answer_en']}{extra}\n\n**Sources**\n{cites}\n\n**Data rows used:** {rows}\n\n**Limits:** {'; '.join(r['limits']) or '—'}"
    return en, r["answer_ur"]


# ── app ─────────────────────────────────────────────────────────────────────

def build_app(snapshot_dir, allow_approvals: bool = False, refresh=None):
    """`refresh` is an optional callable that updates the snapshot folder (the Space downloads it)."""
    import gradio as gr

    s = Snapshot(snapshot_dir)

    def render(audience):
        if refresh:
            try:
                refresh()
            except Exception as e:  # keep showing the last snapshot
                print(f"refresh failed: {e}")
        ap = s.table("approvals.csv")
        pend = ap[ap["status"] == "pending"] if len(ap) and "status" in ap else pd.DataFrame()
        return (product_html(s, audience), share_for(s, audience), status_panel(s), recent_series(s), metrics_table(s),
                s.table("errors_by_group.csv"), s.table("quality_log.parquet").head(60), s.table("local_validation.csv"), pend,
                gr.update(choices=pend["proposal_id"].tolist() if len(pend) else [], value=None))

    def answer(x):
        en, ur = ask(s, x)
        return en, f"<div class='sh-ur' lang='ur'>{html.escape(ur)}</div>"

    with gr.Blocks(title="PeshGo — Lahore air, for the people who work in it") as app:
        gr.HTML(f"<style>{CSS}</style><div class='sh-top'><div class='sh-logo'>PG</div><div>"
                "<h1>PeshGo · پیشگو</h1>"
                "<p>Tomorrow's air in Lahore, for the people who can't stay indoors.</p></div></div>")
        audience = gr.Radio(product.AUDIENCES, value=product.AUDIENCES[0], label="Who is this for?", interactive=True)
        hero = gr.HTML()
        share = gr.Textbox(label="Copy and send (WhatsApp-ready, English + Urdu)", lines=10, interactive=False)
        with gr.Tab("Ask · پوچھیں"):
            q = gr.Textbox(label="Question (English or Urdu)", placeholder="Should I work outside tomorrow morning?  ·  کیا کل صبح باہر کام کرنا ٹھیک ہے؟")
            go = gr.Button("Ask", variant="primary")
            a_en = gr.Markdown()
            a_ur = gr.HTML()
            go.click(answer, q, [a_en, a_ur])
            q.submit(answer, q, [a_en, a_ur])
        with gr.Tab("Evidence"):
            gr.Markdown("**Model against baselines.** Held-out test period, scored once. Baseline (declared before results): next 24 h = mean of the last 24 h.")
            metrics = gr.Dataframe(interactive=False)
            gr.Markdown("**Errors by group** (where the model is weakest)")
            groups = gr.Dataframe(interactive=False)
            gr.Markdown("**Last 14 days**, observed and forecast")
            series = gr.LinePlot(x="time (PKT)", y="PM2.5 µg/m³", color="series", height=340)
            gr.Markdown("**Local validation:** our PM2.5 (as US EPA AQI) against Punjab EPD's daily Lahore AQI")
            local = gr.Dataframe(interactive=False)
        with gr.Tab("Pipeline & data quality"):
            status = gr.HTML()
            refresh_btn = gr.Button("Refresh", size="sm")
            gr.Markdown("**Data Quality Agent decisions** (fix, fill, flag, propose)")
            qlog = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Approvals"):
            gr.Markdown("Risky data decisions wait here for a person (human in the loop)." +
                        ("" if allow_approvals else " **Read-only on the public Space**: decisions are made on the team's local app."))
            pending = gr.Dataframe(interactive=False)
            pick = gr.Dropdown(label="Proposal", choices=[], interactive=allow_approvals)
            who = gr.Textbox(label="Your name", interactive=allow_approvals)
            with gr.Row():
                yes = gr.Button("Approve", interactive=allow_approvals)
                no = gr.Button("Reject", interactive=allow_approvals)
            msg = gr.Markdown()

        outs = [hero, share, status, series, metrics, groups, qlog, local, pending, pick]
        app.load(render, audience, outs)
        refresh_btn.click(render, audience, outs)
        audience.change(lambda a: (product_html(s, a), share_for(s, a)), audience, [hero, share])
        if hasattr(gr, "Timer"):
            gr.Timer(600).tick(render, audience, outs)

        if allow_approvals:
            from . import quality_agent as qa
            from .settings import path
            import shutil

            def decide(pid, name, approve):
                if not pid or not (name or "").strip():
                    return "Pick a proposal and enter your name."
                ok = qa.decide(pid, approve, name.strip())
                src = path("data") / "approvals.csv"
                if src.exists():
                    shutil.copy2(src, s.dir / "approvals.csv")
                return (f"Recorded: {'approved' if approve else 'rejected'} by {name.strip()}. It applies from the next pipeline run."
                        if ok else "That proposal is no longer pending.")
            yes.click(lambda p, w: decide(p, w, True), [pick, who], msg).then(render, audience, outs)
            no.click(lambda p, w: decide(p, w, False), [pick, who], msg).then(render, audience, outs)
    return app


if __name__ == "__main__":
    import argparse

    from .settings import path
    ap = argparse.ArgumentParser(description="Run the PeshGo app locally (reads data/snapshot).")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--read-only", action="store_true", help="hide the approval buttons")
    args = ap.parse_args()
    build_app(path("snapshot"), allow_approvals=not args.read_only).launch(server_name=args.host, server_port=args.port)

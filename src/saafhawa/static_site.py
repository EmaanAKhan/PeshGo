"""PeshGo as an installable phone-style web app, rendered to ONE self-contained HTML file.

Hugging Face charges for Gradio Spaces on the free tier, but static Spaces are free and
always on. Every pipeline run renders this page from the snapshot and uploads it to a
static Space, so the public app updates hourly with no server.

The screen is deliberately small: who it is for (workers, parents, everyone), one big
answer, what to do, when the air was cleaner, and a Share button that stays at the
bottom. One language shows at a time (English or Urdu). Everything an examiner wants
(model against baselines, errors by group, data quality, pipeline status) sits behind a
single "Details" section. Assets written by write_assets() (manifest, icons, service
worker) make it installable on a phone's home screen and keep the last forecast
available offline. The live question box needs Python, so it stays in the local app.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from . import product
from .dashboard import Snapshot, _stale_banner, forecast_numbers, metrics_table, share_for, status_panel, trust_tiles
from .settings import config

KEYS = ["workers", "parents", "everyone"]
SHORT = {"Outdoor workers": "Workers", "Parents & schools": "Parents & schools", "Everyone": "Everyone"}

CSS = """
:root{color-scheme:light dark;--bg:#f2f5f7;--card:#fff;--ink:#16202a;--muted:#5a6672;--line:#e1e6eb;--accent:#0f6b5c;--soft:#edf1f4;--warn:#9a3412;--warnbg:#fff1e6}
@media (prefers-color-scheme:dark){:root{--bg:#0e1317;--card:#182027;--ink:#e8edf2;--muted:#9aa7b3;--line:#2a3540;--accent:#4cc3ae;--soft:#202a33;--warn:#ffb98a;--warnbg:#3a2616}}
*{box-sizing:border-box}[hidden]{display:none !important}
html[data-lang=en] [data-l=ur],html[data-lang=ur] [data-l=en]{display:none}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;-webkit-tap-highlight-color:transparent}
[lang=ur]{direction:rtl;text-align:right;font-family:'Noto Nastaliq Urdu','Jameel Noori Nastaleeq','Segoe UI',serif;line-height:2.1}
.app{max-width:520px;margin:0 auto;padding:10px 14px 130px}
.bar{display:flex;align-items:center;justify-content:space-between;padding:4px 2px 12px}
.brand{display:flex;align-items:center;gap:9px;font-size:19px;font-weight:700}
.brand .ur{font-weight:600;color:var(--muted);font-size:17px}
.logo{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#0f6b5c,#2e9d57);color:#fff;display:grid;place-items:center;font-weight:800;font-size:14px}
.lang{display:flex;gap:4px;background:var(--soft);padding:3px;border-radius:99px}
.lang button{border:0;background:transparent;color:var(--muted);font:inherit;font-size:14px;font-weight:700;padding:5px 12px;border-radius:99px;cursor:pointer}
.lang button[aria-pressed=true]{background:var(--card);color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.15)}
.seg{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;background:var(--soft);padding:4px;border-radius:16px;margin-bottom:12px}
.seg button{border:0;background:transparent;padding:10px 4px;border-radius:12px;font:inherit;font-size:14px;font-weight:700;color:var(--muted);cursor:pointer}
.seg button[aria-pressed=true]{background:var(--card);color:var(--ink);box-shadow:0 1px 4px rgba(0,0,0,.15)}
.card{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:18px;margin-bottom:12px}
.hero{background:var(--c);border:0;padding:22px 20px 20px}.hero,.hero *{color:var(--t)}
.hero .when{font-size:12.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;opacity:.9}
.hero .cat{font-size:36px;font-weight:800;line-height:1.15;letter-spacing:-.02em;margin:6px 0 4px}
.hero .num{font-size:15px;opacity:.95}.hero .num b{font-size:24px;letter-spacing:-.01em}
.act h2{font-size:21px;line-height:1.3;margin:0 0 6px}
.act ul{margin:0;padding-left:20px;font-size:16.5px}.act li{margin:7px 0}
.act [lang=ur] ul{padding-right:22px;padding-left:0}
.chip{display:inline-block;margin-top:10px;background:var(--soft);border-radius:99px;padding:7px 13px;font-size:14px;font-weight:600}
.alert{margin-top:12px;border-radius:14px;padding:10px 13px;background:var(--warnbg);color:var(--warn);font-size:14px;font-weight:600}
.sh-alert{margin-top:12px;border-radius:14px;padding:10px 13px;background:var(--warnbg);color:var(--warn);font-size:14px;font-weight:600}
h3{font-size:12.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
.sh-hours{display:grid;grid-template-columns:repeat(24,1fr);gap:2px;align-items:end}
.sh-hr{display:flex;flex-direction:column;align-items:center;gap:3px}
.sh-hr-b{display:block;width:100%;height:30px;border-radius:5px}
.sh-hr small{font-size:9.5px;color:var(--muted);height:11px}
.sh-best .sh-hr-b{outline:3px solid #14532d;outline-offset:1px}.sh-worst .sh-hr-b{outline:3px solid #7f1d1d;outline-offset:1px}
.sh-note{font-size:13px;color:var(--muted);margin:8px 0 0;line-height:1.5}
.more summary{font-weight:700;cursor:pointer;padding:2px 0;list-style:none;display:flex;justify-content:space-between}
.more summary::after{content:'+';color:var(--muted)}.more[open] summary::after{content:'\\2212'}
.more h4{margin:16px 0 6px;font-size:15px}
.sh-trust{display:grid;grid-template-columns:1fr 1fr;gap:8px}.sh-sec{display:none}
.sh-tile{background:var(--soft);border-radius:14px;padding:10px 12px}.sh-tile b{display:block;font-size:19px;color:var(--accent)}.sh-tile span{font-size:12.5px;color:var(--muted);line-height:1.35;display:block}
.tbl{border-collapse:collapse;width:100%;font-size:13px;margin:4px 0 8px;display:block;overflow-x:auto}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}.tbl th{color:var(--muted);font-weight:600}
table{font-size:13px}.sh-pill{display:inline-block;border-radius:99px;padding:1px 9px;font-size:12px;font-weight:700}
.sh-ok{background:#d8f3e1;color:#14532d}.sh-late{background:#fde2e1;color:#7f1d1d}
.how,footer{color:var(--muted);font-size:13px;line-height:1.55}
a{color:var(--accent)}
.dock{position:fixed;left:0;right:0;bottom:0;padding:22px 14px calc(12px + env(safe-area-inset-bottom));background:linear-gradient(to bottom,transparent,var(--bg) 40%)}
.dock .in{max-width:520px;margin:0 auto;display:flex;gap:8px}
.dock button{flex:1;font:inherit;font-weight:800;border:0;border-radius:16px;padding:15px 10px;cursor:pointer;font-size:16px}
.wa{background:#1fa855;color:#fff}.cp{background:var(--card);color:var(--ink);border:1px solid var(--line) !important}
button:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
"""

JS = """
const KEYS = ['workers', 'parents', 'everyone'];
const $$ = (s) => [...document.querySelectorAll(s)];
function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function setLang(l) {
  document.documentElement.dataset.lang = l;
  $$('[data-lang-btn]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.langBtn === l)));
  store('peshgo-lang', l);
}
function show(i) {
  $$('[data-aud-btn]').forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
  $$('[data-aud-sec]').forEach((s, j) => (s.hidden = i !== j));
  try { history.replaceState(null, '', '#' + KEYS[i]); } catch (e) {}
  store('peshgo-aud', KEYS[i]);
}
$$('[data-lang-btn]').forEach((b) => b.addEventListener('click', () => setLang(b.dataset.langBtn)));
$$('[data-aud-btn]').forEach((b, i) => b.addEventListener('click', () => show(i)));
setLang(load('peshgo-lang') || ((navigator.language || '').toLowerCase().startsWith('ur') ? 'ur' : 'en'));
const fromHash = KEYS.indexOf(location.hash.slice(1));
show(fromHash >= 0 ? fromHash : Math.max(0, KEYS.indexOf(load('peshgo-aud'))));
const shareText = () => { const t = document.querySelector('[data-aud-sec]:not([hidden]) textarea'); return t ? t.value : ''; };
document.getElementById('wa').addEventListener('click', () => window.open('https://wa.me/?text=' + encodeURIComponent(shareText()), '_blank', 'noopener'));
document.getElementById('cp').addEventListener('click', async (e) => {
  const b = e.currentTarget, t = shareText();
  try { await navigator.clipboard.writeText(t); } catch (x) { const a = document.createElement('textarea'); a.value = t; document.body.appendChild(a); a.select(); document.execCommand('copy'); a.remove(); }
  b.textContent = 'Copied'; setTimeout(() => (b.textContent = 'Copy'), 1800);
});
if ('serviceWorker' in navigator) { try { navigator.serviceWorker.register('sw.js'); } catch (e) {} }
"""

SW = """const CACHE = 'peshgo-v1';
self.addEventListener('install', (e) => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then((c) => c.addAll(['./', 'manifest.json', 'icon-192.png'])).catch(() => {})); });
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
// Network first so the forecast is always fresh; the last copy is shown when offline.
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request).then((r) => { const c = r.clone(); caches.open(CACHE).then((x) => x.put(e.request, c)).catch(() => {}); return r; })
    .catch(() => caches.match(e.request).then((r) => r || caches.match('./'))));
});
"""

MANIFEST = {"name": "PeshGo · پیشگو", "short_name": "PeshGo", "description": "Lahore air for the people who work in it",
            "start_url": "./", "scope": "./", "display": "standalone", "orientation": "portrait", "lang": "en",
            "background_color": "#0f6b5c", "theme_color": "#0f6b5c",
            "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                      {"src": "icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}]}


def write_assets(out) -> list[str]:
    """Manifest, service worker and icons that make the page installable. Returns the file names."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "sw.js").write_text(SW, encoding="utf-8")
    names = ["manifest.json", "sw.js"]
    try:
        from PIL import Image, ImageDraw, ImageFont
        for size in (192, 512):
            img = Image.new("RGB", (size, size), "#0f6b5c")
            d = ImageDraw.Draw(img)
            d.ellipse([size * .1, size * .1, size * .9, size * .9], fill="#1f9d6b")
            font = None
            for f in ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"):
                try:
                    font = ImageFont.truetype(f, int(size * .36))
                    break
                except OSError:
                    continue
            font = font or ImageFont.load_default(size=int(size * .36))
            d.text((size / 2, size / 2), "PG", fill="white", font=font, anchor="mm")
            img.save(out / f"icon-{size}.png")
            names.append(f"icon-{size}.png")
    except Exception:  # without Pillow the page still works, it just is not installable
        pass
    return names


def _table(df: pd.DataFrame, empty: str) -> str:
    if df is None or df.empty:
        return f"<p class='sh-note'>{html.escape(empty)}</p>"
    return df.to_html(index=False, classes="tbl", border=0, na_rep="", escape=True)


def _groups(s: Snapshot) -> pd.DataFrame:
    g = s.table("errors_by_group.csv")
    if g.empty or "group" not in g:
        return g
    other = g[g["group"] != "location_id"]
    worst = g[g["group"] == "location_id"].query("n >= 100") if "n" in g else g[g["group"] == "location_id"]
    return pd.concat([other, worst], ignore_index=True)


def _quality(s: Snapshot) -> pd.DataFrame:
    ql = s.table("quality_log.parquet")
    if ql.empty or "run_id" not in ql:
        return ql
    cur = ql[ql["run_id"] == ql.sort_values("logged_utc")["run_id"].iloc[-1]]
    return (cur.groupby(["action", "check_name", "table_name"], as_index=False)["rows_affected"].sum()
            .rename(columns={"rows_affected": "rows"}).sort_values(["action", "check_name"]))


def _section(s: Snapshot, audience: str, key: str, n: dict | None, profile: pd.DataFrame) -> str:
    esc = product.esc
    if n is None:
        return (f"<section data-aud-sec><div class='card'><h2>No forecast yet</h2>"
                f"<p class='sh-note'>The pipeline has not produced one yet. Check back after the next hourly run.</p></div></section>")
    cat = n["category"]
    a = product.advice(audience, cat["name"])
    bw = product.best_and_worst(profile)
    chip = (f"<div class='chip'>☀ <span data-l='en'>Cleanest recently: {product.window_label(bw['best'])}</span>"
            f"<span data-l='ur' lang='ur'>حالیہ دنوں میں سب سے صاف وقت: {product.hour_label(bw['best'][0])} سے {product.hour_label(bw['best'][1])}</span></div>") if bw["best"] else ""
    worse = product.category_for(n["p90"])["name"]
    worse_note = (f"<div class='alert'><span data-l='en'>It could reach “{esc(worse)}”. Plan for that if the work can't move.</span>"
                  f"<span data-l='ur' lang='ur'>یہ «{esc(next(c['name_ur'] for c in config()['categories'] if c['name'] == worse))}» تک جا سکتی ہے۔ اگر کام نہیں ہٹ سکتا تو اس کے مطابق تیاری رکھیں۔</span></div>"
                  if worse != cat["name"] else "")
    en = "".join(f"<li>{esc(b)}</li>" for b in a["bullets_en"])
    ur = "".join(f"<li>{esc(b)}</li>" for b in a["bullets_ur"])
    return f"""<section data-aud-sec aria-label="{esc(audience)}">
<div class="card hero" style="--c:{cat['color']};--t:{product.text_on(cat['color'])}">
  <div class="when">Next 24 hours · Lahore</div>
  <div class="cat"><span data-l="en">{esc(cat['name'])}</span><span data-l="ur" lang="ur">{esc(cat.get('name_ur', ''))}</span></div>
  <div class="num"><b>{n['p50']:.0f}</b> µg/m³ PM2.5 · <span data-l="en">likely {n['p10']:.0f} to {n['p90']:.0f}</span><span data-l="ur" lang="ur">ممکنہ {n['p10']:.0f} سے {n['p90']:.0f}</span></div>
</div>
<div class="card act">
  <h2 data-l="en">{esc(a['headline_en'])}</h2><h2 data-l="ur" lang="ur">{esc(a['headline_ur'])}</h2>
  <ul data-l="en">{en}</ul><ul data-l="ur" lang="ur">{ur}</ul>
  {chip}{worse_note}{_stale_banner(s)}
</div>
<textarea id="share-{key}" hidden aria-hidden="true">{html.escape(share_for(s, audience))}</textarea>
</section>"""


def render(snapshot_dir) -> str:
    """The whole app as one HTML string."""
    s = Snapshot(snapshot_dir)
    built = pd.Timestamp.now(tz=config()["display_timezone"]).strftime("%a %d %b, %H:%M PKT")
    n = forecast_numbers(s)
    profile = product.hourly_profile(s.table("recent_pm25.parquet"))
    seg = "".join(f"<button type='button' data-aud-btn aria-pressed='false'>{html.escape(SHORT[a])}</button>" for a in product.AUDIENCES)
    sections = "".join(_section(s, a, KEYS[i], n, profile) for i, a in enumerate(product.AUDIENCES))
    run = s.status.get("last_run") or {}
    return f"""<!doctype html>
<html lang="en" data-lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0f6b5c"><meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="PeshGo"><meta http-equiv="refresh" content="900">
<link rel="manifest" href="manifest.json"><link rel="apple-touch-icon" href="icon-192.png"><link rel="icon" href="icon-192.png">
<title>PeshGo · Lahore air for the people who work in it</title>
<meta name="description" content="A 24-hour PM2.5 forecast for Lahore with a likely range and cited advice in Urdu and English, for outdoor workers, parents and schools.">
<style>@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');{CSS}</style></head>
<body><div class="app">
<header class="bar"><div class="brand"><span class="logo">PG</span><span>PeshGo</span><span class="ur" lang="ur">پیشگو</span></div>
<div class="lang" role="group" aria-label="Language"><button type="button" data-lang-btn="en" aria-pressed="true">EN</button><button type="button" data-lang-btn="ur" aria-pressed="false">اردو</button></div></header>
<nav class="seg" aria-label="Who is this for?">{seg}</nav>
<main>{sections}</main>
<div class="card"><h3>Hour by hour · last two weeks</h3>{product.hours_strip(profile)}</div>
<details class="card more"><summary>Details and evidence</summary>
<h4>Why you can check this</h4>{trust_tiles(s) or "<p class='sh-note'>Available after the first training.</p>"}
<h4>The evidence: model against the baselines</h4>{_table(metrics_table(s), "The model has not been trained yet.")}
<h4>Where the model is weakest</h4>{_table(_groups(s), "Available after the first training.")}
<h4>What the Data Quality Agent did in the latest run</h4>{_table(_quality(s), "No quality log yet.")}
<h4>Pipeline status and data freshness</h4>{status_panel(s)}
<h4>Checked against Punjab EPD's official daily AQI</h4>{_table(s.table("local_validation.csv"), "No comparison has been run yet.")}
<p class="how"><b>How it works.</b> Every hour a pipeline collects readings from Lahore's monitors, weather, satellite estimates and fire detections, checks and cleans them
(every fix is logged), and a model forecasts the next 24 hours with a likely range. The advice follows the US EPA guide, which lists outdoor workers among the groups most exposed.
The question box (answers with citations, or “I don't know”) runs in the team's local app because it needs a server.</p></details>
<footer>The number is the 24-hour average across Lahore's monitors, not your street. Not medical advice: follow Punjab government and school notices. The Urdu was drafted with AI help;
a native reader should check it. Built {html.escape(built)} · run {html.escape(str(run.get('run_id', 'none yet')))} ({html.escape(str(run.get('status', 'unknown')))}).
Data: OpenAQ, Open-Meteo (ERA5, Copernicus CAMS), NASA FIRMS. Guidance: US EPA AirNow, WHO, Punjab Clean Air Action Plan.</footer>
</div>
<div class="dock"><div class="in"><button type="button" class="wa" id="wa">Share on WhatsApp</button><button type="button" class="cp" id="cp">Copy</button></div></div>
<script>{JS}</script></body></html>"""

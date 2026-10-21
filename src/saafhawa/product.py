"""The product layer: what the forecast means for a person, in words they can act on.

Everything here is derived from the forecast rows and the cited AirNow guidance:
the wording follows the US EPA "Air Quality Guide for Particle Pollution" rows
(Good .. Hazardous). AirNow lists outdoor workers among the people most
exposed, so for "Outdoor workers" the sensitive-group guidance starts one step
earlier in the story, never later. Mask advice is deliberately absent: none of
the documents we cite covers it.

The Urdu was drafted with AI help and should be read by a native speaker before
real-world use; the dashboard says so.
"""

from __future__ import annotations

import html

import pandas as pd

from .settings import config

AUDIENCES = ["Outdoor workers", "Parents & schools", "Everyone"]

# category -> (headline_en, headline_ur, bullets_en, bullets_ur)
_WORKERS = {
    "Good": ("Good air. Work as usual.", "ہوا اچھی ہے۔ معمول کے مطابق کام کریں۔",
             ["No special precautions needed."], ["کسی خاص احتیاط کی ضرورت نہیں۔"]),
    "Moderate": ("Fine for most people. Listen to your body.", "زیادہ تر لوگوں کے لیے ٹھیک ہے۔ اپنے جسم کی سنیں۔",
                 ["Coughing or shortness of breath are signs to take it easier.",
                  "If smoke or dust affects you quickly, keep outdoor work shorter and less intense."],
                 ["کھانسی یا سانس پھولنا اس بات کی علامت ہے کہ کام ہلکا کر دیں۔",
                  "اگر دھوئیں یا گرد سے آپ کو جلدی تکلیف ہوتی ہے تو باہر کا کام مختصر اور ہلکا رکھیں۔"]),
    "Unhealthy for sensitive groups": ("Take more breaks. Outdoor workers are among the most exposed.",
                                       "زیادہ وقفے لیں۔ باہر کام کرنے والے سب سے زیادہ متاثر ہونے والوں میں شامل ہیں۔",
                                       ["Make outdoor work shorter and less intense where you can, and take more breaks.",
                                        "Watch for coughing or shortness of breath.",
                                        "If you have asthma, follow your asthma action plan."],
                                       ["جہاں ممکن ہو باہر کا کام مختصر اور ہلکا رکھیں اور زیادہ وقفے لیں۔",
                                        "کھانسی یا سانس پھولنے پر نظر رکھیں۔",
                                        "اگر آپ کو دمہ ہے تو ڈاکٹر کے بتائے ہوئے طریقے پر عمل کریں۔"]),
    # The AirNow guide lists outdoor workers as a sensitive group, so from "Unhealthy" up
    # they get the sensitive-group guidance directly, not the "everyone else" tier.
    "Unhealthy": ("Outdoor workers are a sensitive group: avoid long or intense outdoor work.",
                  "باہر کام کرنے والے حساس افراد میں شامل ہیں: باہر لمبا یا سخت کام نہ کریں۔",
                  ["Move heavy tasks to a better time, or indoors, if you can.",
                   "If the work cannot move, keep it short and less intense, and take more breaks.",
                   "Watch for coughing or shortness of breath. With heart disease, palpitations or unusual tiredness need a doctor."],
                  ["اگر ممکن ہو تو بھاری کام بہتر وقت پر کریں یا اندر منتقل کریں۔",
                   "اگر کام نہیں ہٹ سکتا تو اسے مختصر اور ہلکا رکھیں اور زیادہ وقفے لیں۔",
                   "کھانسی یا سانس پھولنے پر نظر رکھیں۔ دل کے مریض کو دھڑکن تیز ہونے یا غیر معمولی تھکن پر ڈاکٹر سے رجوع کرنا چاہیے۔"]),
    "Very unhealthy": ("Avoid all physical work outdoors. Reschedule if you can.", "باہر کوئی جسمانی کام نہ کریں۔ ممکن ہو تو وقت بدل لیں۔",
                       ["The US EPA guide lists outdoor workers as a sensitive group, and for sensitive groups it says avoid all physical activity outdoors at this level.",
                        "Reschedule to a time when the air is better, or move the work indoors.",
                        "If you truly cannot, keep it as short as possible, take frequent breaks, and watch for coughing or shortness of breath."],
                       ["امریکی ماحولیاتی ادارے (EPA) کی گائیڈ باہر کام کرنے والوں کو حساس افراد میں شمار کرتی ہے اور اس سطح پر حساس افراد کو باہر کوئی جسمانی سرگرمی نہ کرنے کا کہتی ہے۔",
                        "ایسے وقت پر کام رکھیں جب ہوا بہتر ہو، یا کام اندر منتقل کریں۔",
                        "اگر بالکل ممکن نہ ہو تو کام کم سے کم رکھیں، بار بار وقفہ لیں اور کھانسی یا سانس پھولنے پر نظر رکھیں۔"]),
    "Hazardous": ("Health alert. Everyone should avoid all physical activity outdoors.", "صحت کے لیے خطرہ۔ سب لوگ باہر کوئی جسمانی سرگرمی نہ کریں۔",
                  ["Sensitive groups, including outdoor workers: remain indoors and keep activity low.",
                   "If your work cannot stop, ask your employer for shorter shifts and somewhere safe to rest."],
                  ["حساس افراد، بشمول باہر کام کرنے والے: اندر رہیں اور سرگرمی کم رکھیں۔",
                   "اگر کام روکا نہیں جا سکتا تو مالک سے مختصر شفٹ اور آرام کی محفوظ جگہ مانگیں۔"]),
}

_PARENTS = {
    "Good": ("Good air. Outdoor play and assembly are fine.", "ہوا اچھی ہے۔ باہر کھیل اور اسمبلی ٹھیک ہیں۔",
             ["No special precautions needed."], ["کسی خاص احتیاط کی ضرورت نہیں۔"]),
    "Moderate": ("Fine for most children.", "زیادہ تر بچوں کے لیے ٹھیک ہے۔",
                 ["Children who are unusually sensitive: keep outdoor activity shorter and less intense.",
                  "Watch for coughing or shortness of breath."],
                 ["جو بچے جلدی متاثر ہوتے ہیں ان کی باہر کی سرگرمی مختصر اور ہلکی رکھیں۔", "کھانسی یا سانس پھولنے پر نظر رکھیں۔"]),
    "Unhealthy for sensitive groups": ("Children are a sensitive group today.", "آج بچے حساس افراد میں شامل ہیں۔",
                                       ["Make outdoor activities shorter and less intense, and take more breaks.",
                                        "Consider a short assembly and indoor sports.",
                                        "Children with asthma: follow their asthma action plan."],
                                       ["باہر کی سرگرمیاں مختصر اور ہلکی رکھیں اور زیادہ وقفے لیں۔",
                                        "اسمبلی مختصر کرنے اور کھیل اندر منتقل کرنے پر غور کریں۔",
                                        "دمہ والے بچے ڈاکٹر کے بتائے ہوئے طریقے پر عمل کریں۔"]),
    "Unhealthy": ("Move assembly and sports indoors.", "اسمبلی اور کھیل اندر رکھیں۔",
                  ["Sensitive groups, including children, should avoid long or intense outdoor activities.",
                   "Consider rescheduling or moving activities indoors.",
                   "Everyone else: reduce long or intense activities and take more breaks."],
                  ["بچوں سمیت حساس افراد باہر لمبی یا سخت سرگرمی سے گریز کریں۔",
                   "سرگرمیوں کا وقت بدلنے یا انہیں اندر منتقل کرنے پر غور کریں۔",
                   "باقی سب لمبی یا سخت سرگرمی کم کریں اور زیادہ وقفے لیں۔"]),
    "Very unhealthy": ("Keep children's activities indoors.", "بچوں کی سرگرمیاں اندر رکھیں۔",
                       ["Sensitive groups: avoid all physical activity outdoors.",
                        "Reschedule to a time when the air is better, or move activities indoors."],
                       ["حساس افراد باہر کوئی جسمانی سرگرمی نہ کریں۔",
                        "ایسے وقت پر سرگرمی رکھیں جب ہوا بہتر ہو، یا اسے اندر منتقل کریں۔"]),
    "Hazardous": ("Health alert: no outdoor activity.", "صحت کے لیے خطرہ: باہر کوئی سرگرمی نہیں۔",
                  ["Everyone should avoid all physical activity outdoors.",
                   "Sensitive groups: remain indoors and keep activity low.",
                   "Follow official school notices."],
                  ["سب لوگ باہر کوئی جسمانی سرگرمی نہ کریں۔", "حساس افراد اندر رہیں اور سرگرمی کم رکھیں۔",
                   "اسکول کے سرکاری اعلانات پر عمل کریں۔"]),
}

_EVERYONE = {
    "Good": ("A good air day.", "ہوا اچھی ہے۔", ["It is a great day to be active outside."], ["باہر سرگرم رہنے کے لیے بہترین دن ہے۔"]),
    "Moderate": ("Acceptable air.", "ہوا قابلِ قبول ہے۔",
                 ["Unusually sensitive people: make outdoor activities shorter and less intense."],
                 ["جو لوگ جلدی متاثر ہوتے ہیں وہ باہر کی سرگرمیاں مختصر اور ہلکی رکھیں۔"]),
    "Unhealthy for sensitive groups": ("Sensitive groups should take it easier.", "حساس افراد احتیاط کریں۔",
                                       ["Sensitive groups: make outdoor activities shorter and less intense, and take more breaks."],
                                       ["حساس افراد باہر کی سرگرمیاں مختصر اور ہلکی رکھیں اور زیادہ وقفے لیں۔"]),
    "Unhealthy": ("Everyone: reduce long or intense outdoor activity.", "سب لوگ باہر لمبی یا سخت سرگرمی کم کریں۔",
                  ["Sensitive groups: avoid long or intense outdoor activities.", "Everyone else: take more breaks."],
                  ["حساس افراد باہر لمبی یا سخت سرگرمی نہ کریں۔", "باقی سب زیادہ وقفے لیں۔"]),
    "Very unhealthy": ("Avoid long or intense outdoor activity.", "باہر لمبی یا سخت سرگرمی سے بچیں۔",
                       ["Sensitive groups: avoid all physical activity outdoors.", "Reschedule or move activities indoors."],
                       ["حساس افراد باہر کوئی جسمانی سرگرمی نہ کریں۔", "وقت بدلیں یا سرگرمی اندر منتقل کریں۔"]),
    "Hazardous": ("Health alert for everyone.", "سب کے لیے صحت کا خطرہ۔",
                  ["Avoid all physical activity outdoors.", "Sensitive groups: remain indoors."],
                  ["باہر کوئی جسمانی سرگرمی نہ کریں۔", "حساس افراد اندر رہیں۔"]),
}

_TABLES = {"Outdoor workers": _WORKERS, "Parents & schools": _PARENTS, "Everyone": _EVERYONE}


def advice(audience: str, category: str) -> dict:
    """Headline and bullets (English and Urdu) for an audience and an AQI category."""
    table = _TABLES.get(audience, _EVERYONE)
    he, hu, be, bu = table.get(category, table["Moderate"])
    return {"headline_en": he, "headline_ur": hu, "bullets_en": list(be), "bullets_ur": list(bu)}


def text_on(color: str) -> str:
    """Readable text colour (dark or white) for a background hex colour."""
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#1b1f24" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def category_for(value: float) -> dict:
    for c in config()["categories"]:
        if value <= c["max"]:
            return c
    return config()["categories"][-1]


# ── the hourly pattern (observed, not forecast) ─────────────────────────────

def hourly_profile(recent: pd.DataFrame) -> pd.DataFrame:
    """Median PM2.5 by hour of day (PKT) over the recent window: hour, pm25, days."""
    if recent is None or recent.empty:
        return pd.DataFrame(columns=["hour", "pm25", "days"])
    t = recent["ts_utc"].dt.tz_convert(config()["display_timezone"])
    d = pd.DataFrame({"hour": t.dt.hour, "date": t.dt.date, "pm25": recent["pm25_median"].astype(float)})
    prof = d.groupby("hour").agg(pm25=("pm25", "median"), days=("date", "nunique")).reindex(range(24)).reset_index()
    return prof


def _window(profile: pd.DataFrame, lo: int, hi: int, width: int, best: bool):
    vals = profile.set_index("hour")["pm25"]
    options = []
    for s in range(lo, hi - width + 2):
        hours = list(range(s, s + width))
        if all(h in vals.index and pd.notna(vals[h]) for h in hours):
            options.append((float(vals[hours].mean()), s))
    if not options:
        return None
    _, s = (min if best else max)(options)
    return s, s + width


def best_and_worst(profile: pd.DataFrame, width: int = 3) -> dict:
    """Cleanest working-day window (06:00-20:00) and the worst window overall."""
    return {"best": _window(profile, 6, 20, width, True), "worst": _window(profile, 0, 23, width, False)}


def hour_label(h: int) -> str:
    h = h % 24
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


def window_label(w) -> str:
    return "n/a" if not w else f"{hour_label(w[0])} to {hour_label(w[1])}"


# ── building blocks for the page and the share text ─────────────────────────

def range_bar(p10: float, p50: float, p90: float, color: str) -> str:
    top = max(250.0, p90 * 1.15)
    pct = lambda v: max(0.0, min(100.0, v / top * 100))
    marks = "".join(f"<i class='sh-tick' style='left:{pct(c['max'])}%'></i>" for c in config()["categories"][:-1] if c["max"] < top)
    return (f"<div class='sh-bar' role='img' aria-label='Likely range {p10:.0f} to {p90:.0f}, middle {p50:.0f} micrograms per cubic metre'>"
            f"{marks}<span class='sh-range' style='left:{pct(p10)}%;width:{max(1.5, pct(p90) - pct(p10))}%'></span>"
            f"<b class='sh-mid' style='left:{pct(p50)}%'></b></div>"
            f"<div class='sh-bar-scale'><span>0</span><span>{top:.0f} µg/m³</span></div>")


def hours_strip(profile: pd.DataFrame) -> str:
    if profile.empty or profile["pm25"].isna().all():
        return "<p class='sh-note'>The hourly pattern appears once the monitors have reported.</p>"
    bw = best_and_worst(profile)
    best = set(range(*bw["best"])) if bw["best"] else set()
    worst = set(range(*bw["worst"])) if bw["worst"] else set()
    cells = []
    for _, r in profile.iterrows():
        h = int(r["hour"])
        if pd.isna(r["pm25"]):
            cells.append(f"<div class='sh-hr' title='{hour_label(h)}: no data'><span class='sh-hr-b' style='background:#cbd2d9'></span><small>{h if h % 3 == 0 else ''}</small></div>")
            continue
        c = category_for(float(r["pm25"]))
        cls = " sh-best" if h in best else (" sh-worst" if h in worst else "")
        cells.append(f"<div class='sh-hr{cls}' title='{hour_label(h)}: about {r['pm25']:.0f} µg/m³'>"
                     f"<span class='sh-hr-b' style='background:{c['color']}'></span><small>{h if h % 3 == 0 else ''}</small></div>")
    days = int(profile["days"].max()) if profile["days"].notna().any() else 0
    return (f"<div class='sh-hours' role='img' aria-label='PM2.5 by hour of day over the last {days} days'>{''.join(cells)}</div>"
            f"<p class='sh-note'><b>Cleanest recently:</b> {window_label(bw['best'])} (working hours) · "
            f"<b>Worst recently:</b> {window_label(bw['worst'])}. This is the pattern of the last {days} days, hour by hour "
            f"(Pakistan time). It is <u>not</u> a forecast: the forecast above is a 24-hour average.</p>")


def share_text(audience: str, p10: float, p50: float, p90: float, category: str, when: str, profile: pd.DataFrame | None = None) -> str:
    a = advice(audience, category)
    cat_ur = next((c["name_ur"] for c in config()["categories"] if c["name"] == category), "")
    lines = [f"Lahore air, {when}: {category.upper()} (about {p50:.0f} µg/m³, likely {p10:.0f} to {p90:.0f}).",
             f"For {audience.lower()}: {a['headline_en']}"] + [f"• {b}" for b in a["bullets_en"]]
    if profile is not None and not profile.empty:
        bw = best_and_worst(profile)
        if bw["best"]:
            lines.append(f"Cleanest recent hours: {window_label(bw['best'])}.")
    # No English date inside the Urdu line: mixing left-to-right text into it scrambles the word order.
    lines += ["", f"لاہور کی ہوا: {cat_ur} (تقریباً {p50:.0f} مائیکروگرام فی مکعب میٹر، ممکنہ حد {p10:.0f} سے {p90:.0f})۔",
              a["headline_ur"]] + [f"• {b}" for b in a["bullets_ur"]]
    lines += ["", "Source: US EPA AirNow particle-pollution guide. Information, not medical advice; follow Punjab government notices.",
              "PeshGo · پیشگو"]
    return "\n".join(lines)


def esc(s) -> str:
    return html.escape(str(s))

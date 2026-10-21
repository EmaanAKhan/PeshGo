"""The Insight Agent (retrieval-augmented, citation-first).

Hard rules, enforced in code:
1. No number from a language model. Forecast numbers are read from
   gold_forecasts by code, and the answer stores those row IDs.
2. No claim without a source. Advice is tied to a retrieved document section;
   quoted guidance is extractive (sentences taken from the document).
3. If nothing relevant is retrieved, the answer is "I don't know", plus what is missing.

An optional language model (LLM_API_KEY, OpenAI-compatible) may only reword
the English text. Its output is rejected if it contains any number that is not
in the gold rows or the quoted sources.
"""

from __future__ import annotations

import json
import re
from datetime import timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import manifest, warehouse
from .settings import ROOT, config, path, secrets

MIN_SCORE = 0.08
MIN_SENTENCE_SCORE = 0.12
EMERGENCY_URDU = "سانس لینے میں شدید دشواری یا سینے میں درد ہو تو فوراً 1122 پر کال کریں یا قریبی ہسپتال جائیں۔"
EMERGENCY_EN = "If someone has severe difficulty breathing or chest pain, call Rescue 1122 or go to the nearest hospital now."

# Plain-language versions of the US EPA AQI guidance for PM2.5 (Air Quality
# Guide for Particle Pollution), one per category, with our Urdu translation.
# `query` retrieves the section that supports the advice, which is cited.
ADVICE = {
    "Good": {
        "en": "Air quality is good. Outdoor activities, including school assembly and sports, are fine.",
        "ur": "ہوا کا معیار اچھا ہے۔ اسکول اسمبلی اور کھیل سمیت باہر کی سرگرمیاں ٹھیک ہیں۔",
        "query": "good air quality little or no risk enjoy outdoor activities",
    },
    "Moderate": {
        "en": "Unusually sensitive people should consider reducing long or heavy outdoor exertion. For most children, normal activity is fine.",
        "ur": "غیر معمولی طور پر حساس افراد لمبی یا سخت جسمانی سرگرمی کم کرنے پر غور کریں۔ زیادہ تر بچوں کے لیے معمول کی سرگرمی ٹھیک ہے۔",
        "query": "moderate unusually sensitive people consider reducing prolonged or heavy exertion",
    },
    "Unhealthy for sensitive groups": {
        "en": "Children, older adults and people with asthma, heart or lung disease should reduce long or heavy outdoor exertion. Consider shortening outdoor assembly and moving sports indoors.",
        "ur": "بچے، بزرگ اور دمہ، دل یا پھیپھڑوں کے مریض لمبی یا سخت سرگرمی کم کریں۔ اسمبلی مختصر کرنے اور کھیل اندر منتقل کرنے پر غور کریں۔",
        "query": "sensitive groups children older adults heart lung disease asthma reduce prolonged heavy exertion",
    },
    "Unhealthy": {
        "en": "Sensitive groups should avoid long or heavy outdoor exertion, and everyone else should reduce it. Hold assembly and sports indoors; asthmatic children should keep their inhaler with them.",
        "ur": "حساس افراد باہر لمبی یا سخت سرگرمی سے گریز کریں اور باقی سب اسے کم کریں۔ اسمبلی اور کھیل اندر رکھیں؛ دمہ والے بچے انہیلر ساتھ رکھیں۔",
        "query": "unhealthy everyone may begin to experience health effects sensitive groups avoid prolonged heavy exertion",
    },
    "Very unhealthy": {
        "en": "Sensitive groups should avoid all outdoor physical activity, and everyone else should avoid long or heavy exertion. Keep children indoors with windows closed.",
        "ur": "حساس افراد باہر ہر طرح کی جسمانی سرگرمی سے بچیں اور باقی سب لمبی یا سخت سرگرمی سے گریز کریں۔ بچوں کو کھڑکیاں بند رکھ کر اندر رکھیں۔",
        "query": "very unhealthy health alert avoid all physical activity outdoors sensitive groups",
    },
    "Hazardous": {
        "en": "Everyone should avoid all outdoor physical activity. Stay indoors, keep activity low, and follow official school-closure notices.",
        "ur": "سب لوگ باہر ہر طرح کی جسمانی سرگرمی سے گریز کریں۔ گھر کے اندر رہیں، سرگرمی کم رکھیں اور اسکول بندش کے سرکاری اعلانات پر عمل کریں۔",
        "query": "hazardous emergency conditions everyone avoid all physical activity outdoors remain indoors",
    },
}

# Urdu (and Roman Urdu) terms mapped to English search words, so Urdu questions
# retrieve the English documents.
GLOSSARY = {
    "اسکول": "school children teens outdoor activities", "سکول": "school children teens outdoor activities", "school": "school children teens outdoor activities",
    "bachay": "children teens", "bachon": "children teens", "بچ": "children teens", "child": "children teens", "kid": "children teens",
    "کل": "tomorrow forecast", "kal": "tomorrow forecast", "آج": "today forecast", "aaj": "today forecast",
    "باہر": "outdoor outside activity", "bahar": "outdoor outside activity", "اسمبلی": "assembly outdoor activities", "assembly": "outdoor activities",
    "کھیل": "sports exercise exertion", "ورزش": "exercise exertion", "دمہ": "asthma lung", "سانس": "breathing lung asthma",
    "ماسک": "mask respirator", "mask": "mask respirator", "سموگ": "smog particle pollution pm2.5", "smog": "smog particle pollution pm2.5",
    "دھند": "smog fog particle pollution", "آلودگی": "pollution air quality", "ہوا": "air quality", "صحت": "health effects",
    "بزرگ": "older adults elderly", "حاملہ": "pregnant", "دل": "heart disease", "کھڑکی": "windows indoors", "گھر": "indoors home",
    "فصل": "crop burning agriculture", "جلا": "burning", "حکومت": "government policy action plan", "پنجاب": "punjab",
}
FORECAST_WORDS = re.compile(r"tomorrow|today|forecast|next 24|kal|aaj|کل|آج|school|اسکول|سکول|assembly|اسمبلی|outdoor|bahar|باہر|aqi|level|how bad|کتنی", re.I)
EMERGENCY_WORDS = re.compile(r"can'?t breathe|cannot breathe|chest pain|emergency|saans nahi|سانس نہیں|سینے میں درد|بے ہوش|unconscious", re.I)
URDU = re.compile(r"[؀-ۿ]")
MISLEADING = re.compile(r"\b(good|safe|healthy|harmless|beneficial|fine)\b.*\b(smog|pollution|pm2\.?5|smoke|haze)\b|\b(smog|pollution|pm2\.?5|smoke|haze)\b.*\b(good|safe|healthy|harmless|beneficial)\b", re.I)
JUNK = re.compile(r"EPA-\d|www\.|https?://|^[A-Z][A-Za-z ]{0,40}\?$|February \d{4}|Office of Air|^Air Quality Guide for", re.I)
# Words every section of these documents shares; they carry no relevance signal.
DOMAIN_STOP = set(ENGLISH_STOP_WORDS) | {"particle", "particles", "pollution", "smog", "air", "quality", "pm2", "pm", "aqi", "health", "airnow", "gov", "www", "https", "good", "day"}


# ── ingest ──────────────────────────────────────────────────────────────────

def _sources() -> list[dict]:
    return yaml.safe_load((ROOT / "docs" / "sources.yaml").read_text(encoding="utf-8"))


def fetch_docs() -> int:
    from .sources.http import get
    n = 0
    for s in _sources():
        r = get(s["url"], headers={"User-Agent": "PeshGo-hackathon/1.0 (research; contact via repo)"}, key="docs", per_minute=20)
        ext = "pdf" if s["type"] == "pdf" else "html"
        n += manifest.save("docs", f"{s['publisher']}: {s['title']} <{s['url']}>", f"docs/{s['id']}.{ext}", r.content, licence=s.get("licence")).new
    return n


def _html_sections(html: str) -> list[tuple[str, str]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    sections, title, buf = [], "Introduction", []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            if buf:
                sections.append((title, " ".join(buf)))
            title, buf = text[:120], []
        else:
            buf.append(text)
    if buf:
        sections.append((title, " ".join(buf)))
    return [(t, x) for t, x in sections if len(x) > 80]


def _page_text(page) -> str:
    """Text in reading order, handling two-column pages.

    Words are grouped into lines. A line with a word crossing the middle, or
    with no real gap at the middle, is full width; other lines belong to the
    left or right column. Columns are read left then right, between full-width
    lines, so sentences are not spliced across columns.
    """
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    if not words:
        return ""
    mid = page.width / 2
    lines: dict[int, list] = {}
    for w in words:
        lines.setdefault(round(w["top"] / 3), []).append(w)
    out, left, right = [], [], []

    def flush():
        out.extend(left)
        out.extend(right)
        left.clear()
        right.clear()

    for key in sorted(lines):
        ws = sorted(lines[key], key=lambda w: w["x0"])
        crosses = any(w["x0"] < mid < w["x1"] for w in ws)
        lw = [w for w in ws if w["x1"] <= mid]
        rw = [w for w in ws if w["x0"] >= mid]
        gap = (min(w["x0"] for w in rw) - max(w["x1"] for w in lw)) if lw and rw else 99
        if crosses or gap < 12:
            flush()
            out.append(" ".join(w["text"] for w in ws))
        else:
            if lw:
                left.append(" ".join(w["text"] for w in lw))
            if rw:
                right.append(" ".join(w["text"] for w in rw))
    flush()
    return " ".join(out)


def _table_rows(table) -> list[tuple[str, str]]:
    """Rows of a guidance table: a label in the first cell ("Unhealthy"), an
    optional range on the next row ("(151-200)"), and the guidance text."""
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in table.extract()]
    out = []
    for i, r in enumerate(rows):
        if len(r) < 2 or not r[0] or not any(r[1:]):
            continue
        label = r[0]
        # Labels can span two rows ("Unhealthy for" + "Sensitive Groups"), then the range.
        for nxt in rows[i + 1:i + 3]:
            if nxt and nxt[0] and not any(nxt[1:]):
                label += " " + nxt[0]
            else:
                break
        text = " ".join(c for c in r[1:] if c)
        if len(text) > 20:
            out.append((f"table: {label}", f"{label}: {text}"))
    return out


def _pdf_sections(p) -> list[tuple[str, str]]:
    import pdfplumber
    out = []
    with pdfplumber.open(p) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            tables = page.find_tables()
            rest = page
            for t in tables:
                rows = _table_rows(t)
                if rows:
                    out.extend((f"page {i} {sec}", txt) for sec, txt in rows)
                    rest = rest.outside_bbox(t.bbox)  # keep table cells out of the running text
            text = " ".join(_page_text(rest).split())
            for j, start in enumerate(range(0, len(text), 1100)):
                chunk = text[start:start + 1200]
                if len(chunk) > 80:
                    out.append((f"page {i}" + (f" part {j + 1}" if len(text) > 1200 else ""), chunk))
    return out


def build_docs() -> pd.DataFrame:
    """Raw documents -> gold_docs sections, and the retrieval index."""
    rows = []
    srcs = {s["id"]: s for s in _sources()}
    # A document fetched again with new content is a new raw file; only the
    # latest fetch of each document is indexed, so section IDs stay unique.
    latest = {}
    for f in manifest.files("docs"):
        latest[f.stem.split("__")[0]] = f
    for doc_id, f in latest.items():
        s = srcs.get(doc_id)
        if not s:
            continue
        sections = _pdf_sections(f) if f.suffix == ".pdf" else _html_sections(f.read_text(encoding="utf-8", errors="ignore"))
        for k, (sec, text) in enumerate(sections):
            rows.append({"chunk_id": f"{doc_id}#{k:03d}", "doc_id": doc_id, "title": s["title"], "publisher": s["publisher"],
                         "url": s["url"], "section": sec, "text": text})
    docs = pd.DataFrame(rows, columns=["chunk_id", "doc_id", "title", "publisher", "url", "section", "text"])
    con = warehouse.connect()
    try:
        con.register("_df", docs)
        con.execute("CREATE OR REPLACE TABLE gold_docs AS SELECT * FROM _df")
        con.unregister("_df")
    finally:
        con.close()
    if len(docs):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        mat = vec.fit_transform(docs["section"] + ". " + docs["text"])
        joblib.dump({"vectorizer": vec, "matrix": mat, "chunk_ids": docs["chunk_id"].tolist()}, path("models") / "rag_index.joblib")
    return docs


# ── answer ──────────────────────────────────────────────────────────────────

def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


class InsightAgent:
    def __init__(self, snapshot_dir=None):
        """Reads the warehouse, or (on the Hugging Face Space) a snapshot folder
        holding gold_docs.parquet, rag_index.joblib and forecasts.parquet."""
        from pathlib import Path
        self.snapshot = Path(snapshot_dir) if snapshot_dir else None
        idx = (self.snapshot / "rag_index.joblib") if self.snapshot else path("models") / "rag_index.joblib"
        self.index = joblib.load(idx) if idx.exists() else None
        if self.index is not None and self.snapshot and not (self.snapshot / "gold_docs.parquet").exists():
            self.index = None  # an index without its documents cannot answer anything
        if not self.index:
            self.docs = pd.DataFrame()
        elif self.snapshot:
            f = self.snapshot / "gold_docs.parquet"
            self.docs = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        else:
            self.docs = warehouse.table("gold_docs")
        self.docs = self.docs.set_index("chunk_id") if len(self.docs) else self.docs

    def expand(self, q: str) -> str:
        extra = [v for k, v in GLOSSARY.items() if k.lower() in q.lower()]
        return q + " " + " ".join(extra)

    def retrieve(self, q: str, k: int = 3) -> list[dict]:
        if not self.index:
            return []
        qv = self.index["vectorizer"].transform([self.expand(q)])
        scores = linear_kernel(qv, self.index["matrix"]).ravel()
        order = np.argsort(-scores)[:k]
        out = []
        for i in order:
            if scores[i] < MIN_SCORE:
                break
            cid = self.index["chunk_ids"][i]
            d = self.docs.loc[cid]
            out.append({"chunk_id": cid, "score": round(float(scores[i]), 3), "title": d["title"], "publisher": d["publisher"],
                        "section": d["section"], "url": d["url"], "text": d["text"]})
        return out

    def table_row(self, cat: str) -> list[dict]:
        """The guidance-table row for exactly this AQI category, if a document has one."""
        if self.docs is None or len(self.docs) == 0:
            return []
        want = cat.lower()
        for cid, d in self.docs.iterrows():
            sec = str(d["section"]).lower()
            if "table:" not in sec:
                continue
            label = re.sub(r"\(.*?\)", "", sec.split("table:", 1)[1]).strip()
            if label == want:
                return [{"chunk_id": cid, "score": 1.0, "title": d["title"], "publisher": d["publisher"],
                         "section": d["section"], "url": d["url"], "text": d["text"]}]
        return []

    def _best_sentences(self, q: str, hits: list[dict], n: int = 2) -> list[dict]:
        """Extractive: the sentences of the retrieved sections that best match the question."""
        sents = []
        for h in hits:
            # Sentence ends and PDF bullet points both separate statements.
            for s in re.split(r"(?<=[.!?])\s+|\s*[•▪●]\s*", h["text"]):
                s = s.strip()
                if 40 <= len(s) <= 400 and not JUNK.search(s):
                    sents.append((s, h))
        if not sents:
            return []
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=list(DOMAIN_STOP))
        try:
            sv = vec.fit_transform([s for s, _ in sents])
        except ValueError:
            return []
        scores = linear_kernel(vec.transform([self.expand(q)]), sv).ravel()
        picked, seen = [], set()
        for i in np.argsort(-scores):
            if scores[i] < MIN_SENTENCE_SCORE or len(picked) >= n:
                break
            s, h = sents[i]
            if s in seen:
                continue
            seen.add(s)
            picked.append({"quote": s, "chunk_id": h["chunk_id"], "title": h["title"], "section": h["section"], "url": h["url"]})
        return picked

    def latest_forecast(self) -> pd.DataFrame:
        if self.snapshot:
            f = self.snapshot / "forecasts.parquet"
            if not f.exists():
                return pd.DataFrame()
            fc = pd.read_parquet(f)
            return fc[fc["issued_utc"] == fc["issued_utc"].max()] if len(fc) else fc
        try:
            fc = warehouse.table("gold_forecasts", "SELECT * FROM gold_forecasts WHERE issued_utc = (SELECT max(issued_utc) FROM gold_forecasts)")
        except Exception:
            return pd.DataFrame()
        return fc

    def answer(self, question: str) -> dict:
        q = (question or "").strip()
        urdu = bool(URDU.search(q))
        result = {"question": q, "answer_en": "", "answer_ur": "", "citations": [], "row_ids": [], "numbers_from_rows": {},
                  "grounded": False, "limits": [], "emergency": None}
        if EMERGENCY_WORDS.search(q):
            result["emergency"] = {"en": EMERGENCY_EN, "ur": EMERGENCY_URDU}
        if not q:
            result["answer_en"], result["answer_ur"] = "Please ask a question.", "براہ کرم سوال پوچھیں۔"
            return result

        # A question built on a false premise ("is smog good for me?") is
        # answered with what the documents say about harm, never by matching
        # the word "good" to an unrelated sentence.
        if MISLEADING.search(q):
            q_search = q + " serious health problems effects harmful"
        else:
            q_search = q
        hits = self.retrieve(q_search)
        wants_forecast = bool(FORECAST_WORDS.search(q))
        fc = self.latest_forecast() if wants_forecast else pd.DataFrame()
        if wants_forecast and fc.empty and not hits:
            q_search = q + " children teens outdoor activities sensitive groups reschedule"
            hits = self.retrieve(q_search)

        if not hits and fc.empty:
            missing = "no trusted document covers this question" + ("" if not wants_forecast else ", and no forecast is available yet")
            result["answer_en"] = f"I don't know. {missing[0].upper() + missing[1:]}. Ask about smog, PM2.5, health precautions, or tomorrow's forecast."
            result["answer_ur"] = "مجھے معلوم نہیں۔ اس سوال کے لیے کوئی معتبر دستاویز موجود نہیں۔ سموگ، PM2.5، صحت سے متعلق احتیاط یا کل کی پیش گوئی کے بارے میں پوچھیں۔"
            result["limits"].append(missing)
            return result

        parts_en, parts_ur = [], []
        if wants_forecast and not fc.empty:
            # Numbers come only from these rows.
            p50 = float(np.median(fc["p50"]))
            p10 = float(np.median(fc["p10"]))
            p90 = float(np.median(fc["p90"]))
            cat = _category(p50)
            start = pd.Timestamp(fc["valid_from_utc"].min()).tz_convert(config()["display_timezone"])
            end = pd.Timestamp(fc["valid_to_utc"].max()).tz_convert(config()["display_timezone"])
            nums = {"pm25_p50": round(p50), "pm25_p10": round(p10), "pm25_p90": round(p90), "sensors": int(len(fc))}
            result["numbers_from_rows"] = nums
            result["row_ids"] = fc["forecast_id"].tolist()
            window = f"{start:%a %d %b %H:%M} to {end:%a %d %b %H:%M} PKT"
            parts_en.append(f"Forecast for Lahore, {window}: average PM2.5 about {nums['pm25_p50']} µg/m³ "
                            f"(likely range {nums['pm25_p10']} to {nums['pm25_p90']}), which is \"{cat}\". "
                            f"Median of {nums['sensors']} monitors.")
            cat_ur = next(c["name_ur"] for c in config()["categories"] if c["name"] == cat)
            parts_ur.append(f"لاہور کی پیش گوئی ({window}): اوسط PM2.5 تقریباً {nums['pm25_p50']} مائیکروگرام فی مکعب میٹر "
                            f"(ممکنہ حد {nums['pm25_p10']} سے {nums['pm25_p90']})، یعنی «{cat_ur}»۔")
            adv = ADVICE[cat]
            support = self.table_row(cat) or self.retrieve(adv["query"], k=1)
            if support:
                parts_en.append(adv["en"])
                parts_ur.append(adv["ur"])
                result["citations"].append({k: support[0][k] for k in ("chunk_id", "title", "section", "url")} | {"for": f"advice for {cat}"})
            else:
                result["limits"].append("no retrieved document supports category advice; advice omitted")
            if _category(p90) != cat:
                result["limits"].append(f"the upper end of the range is \"{_category(p90)}\"; plan for the worse case if a child is sensitive")
        elif wants_forecast:
            result["limits"].append("no forecast is available yet (the pipeline has not produced one)")
            parts_en.append("No forecast is available yet, so I cannot say how bad the air will be. General guidance from the documents:")
            parts_ur.append("ابھی کوئی پیش گوئی دستیاب نہیں، اس لیے میں نہیں بتا سکتا کہ ہوا کتنی خراب ہوگی۔ دستاویزات کی عمومی رہنمائی:")

        quotes = self._best_sentences(q_search, hits) if hits else []
        if quotes and MISLEADING.search(q):
            parts_en.append("No. The trusted documents describe particle pollution (smog) as harmful:")
            parts_ur.append("نہیں۔ معتبر دستاویزات کے مطابق سموگ (ذراتی آلودگی) صحت کے لیے نقصان دہ ہے:")
        for qt in quotes:
            parts_en.append(f"“{qt['quote']}” ({qt['title']}, {qt['section']})")
            result["citations"].append({k: qt[k] for k in ("chunk_id", "title", "section", "url")} | {"for": "quoted guidance"})
        if quotes:
            parts_ur.append("متعلقہ رہنمائی (انگریزی ماخذ سے، حوالہ نیچے):")

        if not quotes and fc.empty:
            result["answer_en"] = "I don't know. The retrieved documents do not answer this question."
            result["answer_ur"] = "مجھے معلوم نہیں۔ دستیاب دستاویزات اس سوال کا جواب نہیں دیتیں۔"
            result["limits"].append("retrieved sections were not relevant enough")
            return result

        result["answer_en"] = " ".join(parts_en)
        result["answer_ur"] = " ".join(parts_ur)
        result["grounded"] = True
        result["limits"].append("information, not medical advice; follow Punjab government and school notices")
        result["answer_en"] = self._reword(result["answer_en"], quotes, result["numbers_from_rows"])
        return result

    def _reword(self, text: str, quotes: list[dict], nums: dict) -> str:
        """Optional LLM rewording, rejected if it introduces any new number."""
        s = secrets()
        if not (s.llm_api_key and s.llm_base_url and s.llm_model):
            return text
        try:
            import requests
            r = requests.post(f"{s.llm_base_url.rstrip('/')}/chat/completions", timeout=30,
                              headers={"Authorization": f"Bearer {s.llm_api_key}"},
                              json={"model": s.llm_model, "temperature": 0, "messages": [
                                  {"role": "system", "content": "Rewrite the text to be clear and friendly for a parent. Keep every number, quote and source exactly. Add nothing new."},
                                  {"role": "user", "content": text}]})
            out = r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return text
        allowed = _numbers(text)
        return out if _numbers(out) <= allowed else text


def _category(v: float) -> str:
    from .model import category
    return category(v)

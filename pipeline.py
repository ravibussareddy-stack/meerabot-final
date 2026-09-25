import asyncio
import json
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List

from google import genai
from google.genai import types

from config import GEMINI_API_KEY
from prompts import COMBINED_SYSTEM, COMBINED_USER

logger = logging.getLogger(__name__)

_gemini = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.7-flash"]


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Scores:
    insight_depth:      int = 0
    specificity:        int = 0
    timeliness:         int = 0
    linkedin_potential: int = 0

    @property
    def overall(self) -> float:
        return round(
            self.insight_depth      * 0.35 +
            self.specificity        * 0.25 +
            self.timeliness         * 0.20 +
            self.linkedin_potential * 0.20,
            1,
        )


@dataclass
class Citation:
    title:  str
    source: str
    url:    str
    date:   str = ""


@dataclass
class PipelineResult:
    decision:  str = "DEVELOP"
    reason:    str = ""
    scores:    Scores = field(default_factory=Scores)
    citations: List[Citation] = field(default_factory=list)
    draft:     str = ""
    timed_out: bool = False


# ── Gemini call ───────────────────────────────────────────────────────────────

def _generate_with_fallback(**kwargs):
    last_err = None
    for model in [MODEL] + FALLBACK_MODELS:
        try:
            return _gemini.models.generate_content(model=model, **kwargs)
        except Exception as e:
            logger.warning("Model %s failed: %s", model, str(e)[:80])
            last_err = e
    raise last_err


# ── News fetch (fast, no Gemini) ──────────────────────────────────────────────

_STOPWORDS = {
    "okay","so","the","a","an","and","or","but","is","are","was","were","be",
    "been","have","has","had","do","does","did","will","would","could","should",
    "may","might","can","to","of","in","on","at","for","from","with","by",
    "about","into","that","this","it","we","they","our","their","my","i","me",
    "you","your","came","back","got","get","went","go","make","made","just",
    "very","really","some","same","like","also","then","than","when","there",
    "here","what","which","who","how","why","if","not","no","yes","all",
    "batch","fourteen","thirteen","twelve","eleven","fifteen","sixteen",
    "looking","insights","give","hi","hello","hey",
}

# Sources that consistently produce off-topic results
_JUNK_SOURCES = {"goop", "people", "tmz", "buzzfeed", "cosmopolitan", "allure",
                 "refinery29", "bustle", "popsugar", "glamour", "elle", "vogue"}


def _query_variants(note: str) -> list:
    words = [
        w for w in note.lower().split()
        if len(w) >= 4 and w.isalpha() and w not in _STOPWORDS
    ]
    unique = list(dict.fromkeys(words))
    # Three progressively specific queries
    return [
        "skincare formulation " + " ".join(unique[:6]),
        "beauty ingredient " + " ".join(unique[:4]),
        "cosmetic brand founder " + " ".join(unique[:3]),
    ]


def _rss_fetch(query: str) -> list:
    encoded = urllib.parse.quote(query[:100])
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    results = []
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            root = ET.fromstring(resp.read())
        for item in root.findall(".//item")[:5]:
            title  = (item.findtext("title")   or "").strip()
            source = (item.findtext("source")  or "").strip()
            pub    = (item.findtext("pubDate") or "").strip()
            link   = (item.findtext("link")    or "").strip()
            if not title:
                continue
            # Skip junk sources
            if any(j in source.lower() for j in _JUNK_SOURCES):
                continue
            results.append(Citation(title=title, source=source, url=link, date=pub))
    except Exception as exc:
        logger.warning("RSS fetch failed for %r: %s", query[:40], exc)
    return results


def _google_news_rss(note: str) -> tuple:
    seen_titles = set()
    all_citations = []

    for query in _query_variants(note):
        for c in _rss_fetch(query):
            if c.title not in seen_titles:
                seen_titles.add(c.title)
                all_citations.append(c)
        if len(all_citations) >= 6:
            break

    # Build numbered context for Gemini
    parts = [f"[{i}] {c.title} ({c.source})" for i, c in enumerate(all_citations)]
    return "\n".join(parts), all_citations


async def _fetch_news(note: str) -> tuple:
    return await asyncio.to_thread(_google_news_rss, note)


# ── Single combined Gemini call ───────────────────────────────────────────────

def _gemini_combined(note: str, news_context: str) -> dict:
    resp = _generate_with_fallback(
        contents=COMBINED_USER.format(note=note, news_context=news_context or "None available."),
        config=types.GenerateContentConfig(
            system_instruction=COMBINED_SYSTEM,
            response_mime_type="application/json",
        ),
    )
    return json.loads(resp.text)


async def _run_combined(note: str, news_context: str) -> dict:
    return await asyncio.to_thread(_gemini_combined, note, news_context)


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(note: str) -> PipelineResult:
    result = PipelineResult()

    # Fetch news in parallel (HTTP only — fast)
    news_context, citations = await _fetch_news(note)

    # Single Gemini call: score + draft together, hard 50s timeout
    try:
        raw = await asyncio.wait_for(_run_combined(note, news_context), timeout=50.0)
    except asyncio.TimeoutError:
        result.timed_out = True
        return result
    except Exception as exc:
        logger.error("Combined call failed: %s", exc)
        result.draft = "⚠️ AI model temporarily unavailable. Please resend in a minute."
        return result

    result.decision = raw.get("decision", "DEVELOP")
    result.reason   = raw.get("reason", "")
    result.draft    = raw.get("draft", "").strip()

    s = raw.get("scores", {})
    result.scores = Scores(
        insight_depth=      int(s.get("insight_depth",      5)),
        specificity=        int(s.get("specificity",        5)),
        timeliness=         int(s.get("timeliness",         5)),
        linkedin_potential= int(s.get("linkedin_potential", 5)),
    )

    # Only keep citations Gemini explicitly used in the draft
    used = raw.get("cited_indices", [])
    if isinstance(used, list) and used:
        result.citations = [
            c for i, c in enumerate(citations) if i in used
        ]
    else:
        result.citations = []

    return result

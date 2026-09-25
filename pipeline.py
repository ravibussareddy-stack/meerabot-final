import asyncio
import json
import logging
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List

from google import genai
from google.genai import types

from config import GEMINI_API_KEY
from prompts import COMBINED_SYSTEM, COMBINED_USER

logger = logging.getLogger(__name__)

# Per-request HTTP timeout so a hung Gemini call can't outlive the Vercel function.
_gemini = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=40_000))

MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.7-flash"]

# Own executor: asyncio.run() waits for the *default* executor's threads on exit,
# which would block the timeout reply until a hung call finished.
_EXEC = ThreadPoolExecutor(max_workers=8)

SOURCES_BUDGET_S = 10.0
TOTAL_BUDGET_S = 50.0  # Vercel kills the function at 60s; leave room to send the reply


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


async def _in_thread(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_EXEC, fn, *args)


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
    title:   str
    source:  str
    url:     str
    date:    str = ""
    snippet: str = ""  # brief excerpt shown to Gemini to establish relevance


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
            _log(f"Model {model} failed: {str(e)[:120]}")
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
    # Longer words tend to be the specific ingredients/techniques
    technical = [w for w in unique if len(w) >= 6][:5]
    short = [w for w in unique if len(w) < 6][:3]
    return [
        # Specific ingredient/technique angle — most likely to find a backing source
        "skincare ingredient science " + " ".join(technical[:4]),
        # Broader formulation angle using all key words
        "skincare formulation " + " ".join(unique[:5]),
        # Mix technical + short words to catch different phrasings
        "beauty skincare " + " ".join((technical[:2] + short)[:4]),
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


# Priority-ordered skincare/formulation terms. Lower number = more specific = searched first.
_TERM_PRIORITY: dict = {
    # Tier 0 — specific ingredients/chemicals
    "dimethicone": 0, "silicone": 0, "retinol": 0, "niacinamide": 0,
    "hyaluronic": 0, "ceramide": 0, "glycolic": 0, "salicylic": 0,
    "ascorbic": 0, "tretinoin": 0, "benzoyl": 0, "squalane": 0,
    "bakuchiol": 0, "azelaic": 0, "lactic": 0, "panthenol": 0,
    "centella": 0, "tranexamic": 0, "kojic": 0, "arbutin": 0,
    "caffeine": 0, "peptide": 0, "collagen": 0, "elastin": 0,
    # Tier 1 — formulation concepts
    "occlusive": 1, "emollient": 1, "humectant": 1, "surfactant": 1,
    "emulsifier": 1, "preservative": 1, "formulation": 1, "keratin": 1,
    "lipid": 1, "sebum": 1, "melanin": 1, "vitamin": 1,
    # Tier 2 — general skincare terms
    "serum": 2, "moisturiser": 2, "moisturizer": 2, "cleanser": 2,
    "sunscreen": 2, "exfoliant": 2, "toner": 2, "actives": 2,
    "barrier": 2, "absorption": 2, "layering": 2, "penetration": 2,
    "ingredient": 2, "bioavailability": 2,
}

# Wikipedia query suffix per tier — more specific terms need broader context to find their page
_TIER_SUFFIX = {0: " skincare", 1: " skincare", 2: " skin care routine"}


def _wikipedia_terms(note: str) -> list:
    """Pick up to 3 skincare terms from the note, highest-priority first, as Wikipedia queries."""
    # Build token set — also split hyphenated words so "silicone-based" → {"silicone", "based"}
    tokens: set = set()
    for w in note.split():
        clean = w.lower().strip("'\".,;:—-!()?")
        tokens.add(clean)
        for part in clean.split("-"):
            if len(part) >= 4:
                tokens.add(part)

    found = {t: p for t, p in _TERM_PRIORITY.items() if t in tokens}
    # Sort: lowest priority tier first, then longer = more specific
    ranked = sorted(found.keys(), key=lambda t: (_TERM_PRIORITY[t], -len(t)))

    # Fallback: longest alpha words not already covered
    if len(ranked) < 3:
        extra = [
            w.lower().strip("'\".,;:—-") for w in note.split()
            if len(w.strip("'\".,;:—-")) >= 7
            and w.strip("'\".,;:—-").isalpha()
            and w.strip("'\".,;:—-").lower() not in _STOPWORDS
            and w.strip("'\".,;:—-").lower() not in _TERM_PRIORITY
        ]
        ranked += [w for w in dict.fromkeys(extra) if w not in ranked]

    return [
        t + _TIER_SUFFIX.get(_TERM_PRIORITY.get(t, 2), " skincare")
        for t in ranked[:3]
    ]


def _wikipedia_search(terms: list) -> list:
    """Search Wikipedia; return articles with snippets so Gemini can judge relevance."""
    results = []
    seen: set = set()
    for term in terms[:3]:
        encoded = urllib.parse.quote(term)
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}"
            f"&format=json&srlimit=2&srnamespace=0"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MeeraBot/1.0 (skincare-linkedin-bot)"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "").strip()
                # Skip disambiguation and articles that are clearly off-topic medical topics
                _WIKI_SKIP = {"scar", "wound", "surgery", "drug", "medication", "disease"}
                if not title or title in seen or "(disambiguation)" in title:
                    continue
                if any(s in title.lower() for s in _WIKI_SKIP):
                    continue
                seen.add(title)
                page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                raw = item.get("snippet", "")
                snippet = raw.replace('<span class="searchmatch">', "").replace("</span>", "")[:160].strip()
                results.append(Citation(title=title, source="Wikipedia", url=page_url, date="", snippet=snippet))
        except Exception as exc:
            logger.warning("Wikipedia search failed for %r: %s", term, exc)
    return results


async def _fetch_all_sources(note: str) -> tuple:
    """Fetch news and Wikipedia in parallel so neither waits on the other."""
    def fetch_news():
        seen_titles: set = set()
        results = []
        for query in _query_variants(note):
            for c in _rss_fetch(query):
                if c.title not in seen_titles:
                    seen_titles.add(c.title)
                    results.append(c)
            if len(results) >= 5:
                break
        return results

    def fetch_wiki():
        return _wikipedia_search(_wikipedia_terms(note))

    async def bounded(fn):
        try:
            return await asyncio.wait_for(_in_thread(fn), timeout=SOURCES_BUDGET_S)
        except Exception as exc:
            _log(f"source fetch dropped ({fn.__name__}): {type(exc).__name__}")
            return []

    news_results, wiki_results = await asyncio.gather(bounded(fetch_news), bounded(fetch_wiki))

    seen_titles: set = {c.title for c in news_results}
    all_citations = list(news_results)
    for c in wiki_results:
        if c.title not in seen_titles:
            seen_titles.add(c.title)
            all_citations.append(c)

    # Build numbered context for Gemini — include snippet so it can judge relevance
    parts = []
    for i, c in enumerate(all_citations):
        line = f"[{i}] {c.title} ({c.source})"
        if c.snippet:
            line += f" — {c.snippet}"
        parts.append(line)

    _log(f"Sources: {len(news_results)} news, {len(wiki_results)} wiki, {len(all_citations)} total")
    return "\n".join(parts), all_citations


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


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(note: str) -> PipelineResult:
    result = PipelineResult()
    start = time.monotonic()

    news_context, citations = await _fetch_all_sources(note)
    _log(f"sources took {time.monotonic() - start:.1f}s")

    remaining = max(5.0, TOTAL_BUDGET_S - (time.monotonic() - start))
    try:
        raw = await asyncio.wait_for(_in_thread(_gemini_combined, note, news_context), timeout=remaining)
    except asyncio.TimeoutError:
        _log(f"gemini timed out after {time.monotonic() - start:.1f}s total")
        result.timed_out = True
        return result
    except Exception as exc:
        _log(f"Combined call failed: {str(exc)[:200]}")
        if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
            result.draft = ("⚠️ Daily AI quota reached (free tier: 20 notes/day). "
                            "It resets at midnight Pacific time — resend then.")
        else:
            result.draft = "⚠️ AI model temporarily unavailable. Please resend in a minute."
        return result
    _log(f"gemini done at {time.monotonic() - start:.1f}s total")

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

    # Gemini-decided citations (works well for news articles)
    used = set(raw.get("cited_indices", []) or [])
    gemini_cited = {i for i in used if isinstance(i, int) and 0 <= i < len(citations)}

    # Auto-cite Wikipedia articles whose snippet shares a known technical term with the draft.
    # Gemini consistently skips citing its own knowledge — bypass that for encyclopedic sources.
    draft_lower = result.draft.lower()
    draft_terms = {t for t in _TERM_PRIORITY if t in draft_lower}
    auto_wiki = set()
    if draft_terms:
        for i, c in enumerate(citations):
            if c.source == "Wikipedia" and c.snippet:
                snippet_lower = c.snippet.lower()
                if any(t in snippet_lower for t in draft_terms):
                    auto_wiki.add(i)

    final_indices = gemini_cited | auto_wiki
    result.citations = [c for i, c in enumerate(citations) if i in final_indices]

    return result

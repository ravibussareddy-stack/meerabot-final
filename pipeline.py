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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
    kind:    str = "news"  # "research" | "news" | "reference"


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
    last_err = quota_err = None
    for model in [MODEL] + FALLBACK_MODELS:
        try:
            return _gemini.models.generate_content(model=model, **kwargs)
        except Exception as e:
            _log(f"Model {model} failed: {str(e)[:120]}")
            last_err = e
            if "RESOURCE_EXHAUSTED" in str(e):
                quota_err = e
    # Quota is the actionable cause even if a later fallback was merely overloaded
    raise quota_err or last_err


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

# Celebrity/lifestyle outlets that produced off-topic matches in testing
_JUNK_SOURCES = {"goop", "people", "tmz", "buzzfeed", "popsugar", "us weekly", "page six"}

# Cosmetic-industry trade press, searched in addition to general beauty news
_TRADE_SITES = (
    "cosmeticsdesign.com", "cosmeticsdesign-europe.com", "cosmeticsbusiness.com", "happi.com",
    "dermatologytimes.com", "personalcareinsights.com", "premiumbeautynews.com",
    "cosmeticsandtoiletries.com", "beautyindependent.com", "glossy.co",
)
_NEWS_MAX_AGE_YEARS = 6


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
    "oxidation": 1, "oxidised": 1, "oxidized": 1, "stability": 1,
    "packaging": 2, "irritation": 2,
    # Tier 2 — general skincare terms
    "serum": 2, "moisturiser": 2, "moisturizer": 2, "cleanser": 2,
    "sunscreen": 2, "exfoliant": 2, "toner": 2, "actives": 2,
    "barrier": 2, "absorption": 2, "layering": 2, "penetration": 2,
    "ingredient": 2, "bioavailability": 2,
}

# Generic chemistry terms whose own Wikipedia page is industrial — search the skincare form instead.
_WIKI_QUERY_ALIASES = {
    "silicone": "dimethicone skin",
    "dimethicone": "dimethicone skin",
    "occlusive": "occlusive moisturizer skin",
    "lipid": "skin lipid barrier",
    "barrier": "skin barrier stratum corneum",
}

# "skin" alone also matches medical/anthropology pages (Scurvy, Light skin), so require
# cosmetic context in the article's opening, or a section that is specifically about skin care.
_COSMETIC_WORDS = ("cosmetic", "skincare", "skin care", "personal care", "topical")
_SECTION_WORDS = _COSMETIC_WORDS + ("aging", "ageing")


def _wikipedia_terms(note: str) -> list:
    """Up to 3 Wikipedia queries built from the note's highest-priority skincare terms."""
    return [_WIKI_QUERY_ALIASES.get(t, t + " skin") for t in _ranked_terms(note)[:3]]


def _ranked_terms(note: str) -> list:
    """Skincare terms in the note, most specific first, with long-word fallbacks."""
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

    return ranked


# ── Relevance scoring (shared by all source types) ────────────────────────────

# Words that mean the same thing in sources (a "silicone" post is backed by a "dimethicone" paper)
_TERM_SYNONYMS = {
    "silicone": ("silicone", "dimethicone", "siloxane"),
    "dimethicone": ("dimethicone", "silicone", "siloxane"),
    "occlusive": ("occlusive", "occlusion", "petrolatum"),
    "moisturiser": ("moisturiser", "moisturizer", "emollient"),
    "moisturizer": ("moisturizer", "moisturiser", "emollient"),
    "ascorbic": ("ascorbic", "ascorbyl"),
    "glycolic": ("glycolic", "aha", "alpha hydroxy", "exfoliat"),
    "lactic": ("lactic", "aha", "alpha hydroxy"),
    "salicylic": ("salicylic", "bha", "beta hydroxy"),
    "oxidation": ("oxidation", "oxidis", "oxidiz", "antioxidant"),
    "oxidised": ("oxidation", "oxidis", "oxidiz", "antioxidant"),
    "oxidized": ("oxidation", "oxidis", "oxidiz", "antioxidant"),
    "irritation": ("irritation", "irritat", "sensitis", "sensitiz"),
    "niacinamide": ("niacinamide", "nicotinamide"),
    "retinol": ("retinol", "retinoid"),
    "barrier": ("barrier", "stratum corneum"),
    "absorption": ("absorption", "penetration", "permeation"),
    "penetration": ("penetration", "permeation", "absorption"),
    "layering": ("layering", "application order"),
}


def _stem(word: str) -> str:
    return word[: max(5, len(word) - 3)]


def _term_groups(terms) -> list:
    """[(weight, (stem, stem, ...)), ...] — one group per concept, weighted by specificity."""
    groups = []
    for t in dict.fromkeys(terms):
        weight = 2 if _TERM_PRIORITY.get(t, 2) <= 1 else 1
        groups.append((weight, tuple(_stem(s) for s in _TERM_SYNONYMS.get(t, (t,)))))
    return groups


def _match(text: str, groups: list) -> tuple:
    """(weighted score, number of distinct concepts matched)."""
    t = text.lower()
    hits = [w for w, stems in groups if any(s in t for s in stems)]
    return sum(hits), len(hits)


def _relevance(text: str, groups: list) -> int:
    return _match(text, groups)[0]


def _search_phrase(terms: list, n: int = 2) -> str:
    return " ".join(terms[:n])


# ── Peer-reviewed research (Europe PMC, includes PubMed) ──────────────────────

def _research_search(note: str) -> list:
    terms = [t for t in _ranked_terms(note) if t in _TERM_PRIORITY][:3]
    if not terms:
        return []

    def clause(t):
        return "(" + " OR ".join(f'"{s}"' for s in _TERM_SYNONYMS.get(t, (t,))) + ")"

    context = '(skin OR cosmetic OR topical OR dermal)'
    results = []
    # Most specific first: all top terms together, then fewer until something matches
    for k in (min(3, len(terms)), 2, 1):
        if k > len(terms):
            continue
        q = " AND ".join(f"TITLE_ABS:{clause(t)}" for t in terms[:k])
        q += f" AND TITLE_ABS:{context} AND PUB_YEAR:[2010 TO 2026]"
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
            {"query": q, "format": "json", "pageSize": 8, "resultType": "core"})
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _WIKI_UA})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            _log(f"Europe PMC failed: {exc}")
            return results
        for r in data.get("resultList", {}).get("result", []):
            title = (r.get("title") or "").strip().rstrip(".")
            if not title:
                continue
            doi = r.get("doi")
            link = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}"
            journal = r.get("journalInfo", {}).get("journal", {}).get("title") or r.get("journalTitle", "")
            results.append(Citation(
                title=title, source=journal, url=link, date=str(r.get("pubYear", "")),
                snippet=(r.get("abstractText") or "")[:600], kind="research"))
        if results:
            break
    return results


# ── News: cosmetic trade press + general beauty news (Google News RSS) ────────

def _rss_fetch(query: str, limit: int = 10) -> list:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en", "gl": "US", "ceid": "US:en"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    results = []
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            root = ET.fromstring(resp.read())
    except Exception as exc:
        _log(f"RSS fetch failed for {query[:40]!r}: {exc}")
        return results
    cutoff = datetime.now(timezone.utc).year - _NEWS_MAX_AGE_YEARS
    for item in root.findall(".//item")[:limit]:
        title  = (item.findtext("title")   or "").strip()
        source = (item.findtext("source")  or "").strip()
        link   = (item.findtext("link")    or "").strip()
        if not title or any(j in source.lower() for j in _JUNK_SOURCES):
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except Exception:
            continue
        if published.year < cutoff:
            continue
        # Google appends " - Source Name" to every headline
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)]
        results.append(Citation(title=title, source=source, url=link,
                                date=published.strftime("%b %Y"), kind="news"))
    return results


def _news_search(note: str) -> list:
    terms = _ranked_terms(note)
    if not terms:
        return []
    phrase = _search_phrase(terms)
    sites = " OR ".join(f"site:{s}" for s in _TRADE_SITES)
    queries = [
        f"{phrase} ({sites})",           # trade press
        f"{phrase} skincare",            # general beauty news
        f"{_search_phrase(terms, 1)} skincare ingredient",
    ]
    seen, out = set(), []
    for q in queries:
        for c in _rss_fetch(q):
            if c.title.lower() not in seen:
                seen.add(c.title.lower())
                out.append(c)
    return out


_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_SKIP = ("scar", "wound", "surgery", "drug", "medication", "disease", "implant", "bleaching")
# Wikimedia asks API clients to identify themselves with a contact URL
_WIKI_UA = "MeeraBot/1.0 (https://github.com/ravibussareddy-stack/meerabot-final)"


def _wiki_get(params: dict) -> dict:
    url = _WIKI_API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_UA})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _is_cosmetic_lead(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _COSMETIC_WORDS)


def _is_skincare_section(section: str) -> bool:
    s = section.lower().strip()
    return s == "skin" or any(w in s for w in _SECTION_WORDS)


def _wikipedia_search(queries: list) -> list:
    """Return only skincare-relevant Wikipedia sources, linking to the skin section when the
    article as a whole is about something else (e.g. Polydimethylsiloxane → #Skin)."""
    candidates, seen = [], set()
    for q in queries[:3]:
        try:
            data = _wiki_get({"action": "query", "list": "search", "srsearch": q, "srlimit": 3,
                              "srnamespace": 0, "srprop": "snippet|sectiontitle"})
        except Exception as exc:
            _log(f"Wikipedia search failed for {q!r}: {exc}")
            continue
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "").strip()
            if not title or title in seen or "(disambiguation)" in title:
                continue
            if any(s in title.lower() for s in _WIKI_SKIP):
                continue
            seen.add(title)
            snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
            candidates.append((title, item.get("sectiontitle", ""), snippet[:160].strip()))

    if not candidates:
        return []

    # One call for every candidate's opening paragraph, to judge what the article is about
    leads = {}
    try:
        data = _wiki_get({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                          "exlimit": "max", "redirects": 1, "titles": "|".join(t for t, _, _ in candidates)})
        for page in data.get("query", {}).get("pages", {}).values():
            leads[page.get("title", "")] = page.get("extract", "")[:800]
    except Exception as exc:
        _log(f"Wikipedia extracts failed: {exc}")

    results = []
    for title, section, snippet in candidates:
        base = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        if section and _is_skincare_section(section):
            url = base + "#" + urllib.parse.quote(section.replace(" ", "_"))
            label = f"{title} — {section} section"
        elif _is_cosmetic_lead(leads.get(title, "")):
            url, label = base, title
        else:
            continue  # article isn't about skin, and the match wasn't in a skin section
        results.append(Citation(title=label, source="Wikipedia", url=url, date="", snippet=snippet,
                                kind="reference"))
    return results


async def _fetch_all_sources(note: str) -> tuple:
    """Fetch research, news and Wikipedia in parallel so none waits on the others."""
    def fetch_news():
        return _news_search(note)

    def fetch_research():
        return _research_search(note)

    def fetch_wiki():
        return _wikipedia_search(_wikipedia_terms(note))

    async def bounded(fn):
        try:
            return await asyncio.wait_for(_in_thread(fn), timeout=SOURCES_BUDGET_S)
        except Exception as exc:
            _log(f"source fetch dropped ({fn.__name__}): {type(exc).__name__}")
            return []

    research, news, wiki = await asyncio.gather(
        bounded(fetch_research), bounded(fetch_news), bounded(fetch_wiki))

    # Pre-rank against the note so Gemini sees the most relevant candidates first
    groups = _term_groups(_ranked_terms(note))
    all_citations = []
    for batch, keep in ((research, 5), (news, 6), (wiki, 4)):
        ranked = sorted(batch, key=lambda c: _relevance(c.title + " " + c.snippet, groups), reverse=True)
        all_citations += ranked[:keep]

    parts = []
    for i, c in enumerate(all_citations):
        line = f"[{i}] ({c.kind}) {c.title} ({c.source} {c.date})".rstrip()
        if c.snippet:
            line += f" — {c.snippet[:200]}"
        parts.append(line)

    _log(f"Sources: {len(research)} research, {len(news)} news, {len(wiki)} wiki")
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

    result.citations = _select_citations(note, result.draft, citations)
    return result


_MAX_PER_KIND = 2


def _qualifies(c: Citation, groups: list) -> bool:
    """A single shared word ("silicone") is too ambiguous — silicone patches, shampoo — so
    headlines must hit two concepts. Papers must be *about* a concept (title), and overlap
    with the post on at least two (title + abstract)."""
    _, title_hits = _match(c.title, groups)
    if c.kind == "news":
        return title_hits >= 2
    if c.kind == "research":
        score, hits = _match(c.title + " " + c.snippet, groups)
        return title_hits >= 1 and hits >= 2 and score >= 4
    return title_hits >= 1  # reference: article/section must be named for a concept


def _select_citations(note: str, draft: str, citations: list) -> list:
    """Pick the sources that actually back this draft. Done deterministically: Gemini tends
    to leave citations empty when the claims come from its own knowledge."""
    draft_lower = draft.lower()
    groups = _term_groups(_ranked_terms(note) + [t for t in _TERM_PRIORITY if t in draft_lower])
    picked = []
    for kind in ("research", "news", "reference"):
        ok = [c for c in citations if c.kind == kind and _qualifies(c, groups)]
        ok.sort(key=lambda c: (_match(c.title, groups)[0], _relevance(c.title + " " + c.snippet, groups)),
                reverse=True)
        picked += ok[:_MAX_PER_KIND]
    return picked

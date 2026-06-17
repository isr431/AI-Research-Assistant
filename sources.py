"""Source registry, deduplication, citation formatting, and validation."""

from __future__ import annotations

import re
import sys
from datetime import date
from typing import Any

from text_utils import tokenize_terms


def _tokenize(text: str) -> set[str]:
    """Return search-relevant lowercase terms from free text."""
    return tokenize_terms(text)


def _overlap_score(terms: set[str], text: str) -> float:
    if not terms:
        return 0.0
    matches = _tokenize(text) & terms
    return len(matches) / len(terms)


def _freshness_score(date_text: str, current_year: int) -> float:
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", date_text)]
    if not years:
        return 0.0
    age = current_year - max(years)
    if age <= 0:
        return 1.0
    if age == 1:
        return 0.6
    if age <= 3:
        return 0.3
    return 0.0


def _primary_source_score(domain: str, terms: set[str]) -> float:
    domain = domain.lower().removeprefix("www.")
    score = 0.0
    if domain.endswith((".edu", ".gov")):
        score += 0.5
    if domain.startswith(("developer.", "docs.", "newsroom.")):
        score += 0.4
    base = domain.split(".")[0].replace("-", " ")
    if _tokenize(base) & terms:
        score += 0.7
    return score


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip protocol, www prefix, trailing slash, and tracking params.

    Used as a dedup key so that equivalent URLs collapse into one source.
    """
    url = re.sub(r"^https?://(www\.)?", "", url)
    url = url.rstrip("/")
    # Remove common tracking / referral parameters
    url = re.sub(r"[?&](utm_\w+|ref|source|fbclid|gclid)=[^&]*", "", url)
    # Fix query string separators if a parameter at the start was removed
    url = url.replace("?&", "?")
    if "?" not in url and "&" in url:
        url = url.replace("&", "?", 1)
    # Clean up leftover ? or & at the end
    url = url.rstrip("?&")
    return url


# ---------------------------------------------------------------------------
# Source Registry
# ---------------------------------------------------------------------------

class SourceRegistry:
    """Assigns incrementing IDs to unique sources and merges duplicate snippets."""

    def __init__(self) -> None:
        self._next_id: int = 1
        # normalized_url → source_id
        self._url_map: dict[str, int] = {}
        # source_id → source dict
        self._sources: dict[int, dict[str, Any]] = {}

    # -- public API ----------------------------------------------------------

    def add(self, result: dict[str, Any]) -> int:
        """Register a search result.  Returns its (possibly existing) source ID.

        ``result`` is expected to have keys: url, title, domain, snippets, date,
        query_origin.
        """
        norm = normalize_url(result["url"])

        if norm in self._url_map:
            sid = self._url_map[norm]
            # Merge new snippets that we haven't seen yet
            existing_snippets = set(self._sources[sid]["snippets"])
            for snippet in result.get("snippets", []):
                if snippet not in existing_snippets:
                    self._sources[sid]["snippets"].append(snippet)
                    existing_snippets.add(snippet)
            query_origin = result.get("query_origin", "")
            if query_origin and query_origin not in self._sources[sid]["query_origins"]:
                self._sources[sid]["query_origins"].append(query_origin)
            return sid

        sid = self._next_id
        self._next_id += 1
        self._url_map[norm] = sid
        self._sources[sid] = {
            "url": result["url"],
            "title": result.get("title", "Untitled"),
            "domain": result.get("domain", ""),
            "date": result.get("date", ""),
            "snippets": list(result.get("snippets", [])),
            "query_origins": [result["query_origin"]]
            if result.get("query_origin")
            else [],
            "query_index": result.get("query_index"),
            "result_rank": result.get("result_rank"),
        }
        return sid

    def get(self, source_id: int) -> dict[str, Any] | None:
        return self._sources.get(source_id)

    def all(self) -> dict[int, dict[str, Any]]:
        """Return a copy of the full registry (id → source dict)."""
        return dict(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def _ranked_source_ids(self) -> list[int]:
        return sorted(
            self._sources,
            key=lambda sid: (
                -float(self._sources[sid].get("source_score", 0.0)),
                self._sources[sid].get("query_index")
                if self._sources[sid].get("query_index") is not None
                else 999,
                self._sources[sid].get("result_rank")
                if self._sources[sid].get("result_rank") is not None
                else 999,
                sid,
            ),
        )

    def top_sources_for_fetch(self, limit: int) -> list[dict[str, Any]]:
        """Return top-scored source summaries that have not been fetched yet."""
        selected: list[dict[str, Any]] = []
        for sid in self._ranked_source_ids():
            src = self._sources[sid]
            if src.get("page_excerpt") or src.get("page_fetch_status"):
                continue
            selected.append({
                "id": sid,
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "domain": src.get("domain", ""),
            })
            if len(selected) >= limit:
                break
        return selected

    def set_page_excerpt(self, source_id: int, excerpt: str, status: str) -> None:
        """Attach a fetched page excerpt or fetch status to a source."""
        src = self._sources.get(source_id)
        if not src:
            return
        src["page_fetch_status"] = status
        if excerpt:
            src["page_excerpt"] = excerpt

    def score_sources(
        self,
        question: str,
        queries: list[str] | None = None,
        current_date: str | None = None,
    ) -> None:
        """Assign lightweight relevance scores used for context ordering."""
        current_year = date.today().year
        if current_date:
            match = re.search(r"\b(20\d{2})\b", current_date)
            if match:
                current_year = int(match.group(1))

        all_origins = [
            origin
            for src in self._sources.values()
            for origin in src.get("query_origins", [])
        ]
        query_text = " ".join([question, *(queries or []), *all_origins])
        terms = _tokenize(query_text)

        for src in self._sources.values():
            snippets_text = " ".join(src.get("snippets", []))
            origins_text = " ".join(src.get("query_origins", []))
            result_rank = src.get("result_rank") or 99
            rank_score = max(0.0, 1.0 - ((result_rank - 1) * 0.1))

            score = 0.0
            score += 4.0 * _overlap_score(terms, src.get("title", ""))
            score += 2.5 * _overlap_score(terms, snippets_text)
            score += 1.0 * _overlap_score(terms, origins_text)
            score += 0.8 * _freshness_score(src.get("date", ""), current_year)
            score += 0.8 * _primary_source_score(src.get("domain", ""), terms)
            score += 0.5 * rank_score

            src["source_score"] = round(score, 4)

    # -- formatting ----------------------------------------------------------

    def format_knowledge_context(self, max_tokens: int = 40_000) -> str:
        """Build the tagged context string for the LLM.

        Each source is labelled with its source index and key metadata so the
        LLM can cite correctly.  If the total exceeds *max_tokens* (estimated as
        ``len(text) // 4``), the lowest-ranked sources are truncated first.
        """
        blocks: list[str] = []
        source_ids = self._ranked_source_ids()
        for sid in source_ids:
            src = self._sources[sid]
            lines = [
                f"[Source {sid}]",
                f"Title: {src['title']}",
                f"URL: {src['url']}",
            ]
            if src.get("domain"):
                lines.append(f"Domain: {src['domain']}")
            if src.get("date"):
                lines.append(f"Date: {src['date']}")
            if src.get("query_origins"):
                queries = "; ".join(src["query_origins"])
                lines.append(f"Found by query: {queries}")
            lines.append("Snippets:")
            snippets = src.get("snippets", [])
            if snippets:
                lines.extend(f'- "{snippet}"' for snippet in snippets)
            else:
                lines.append("- No snippet provided.")
            if src.get("page_excerpt"):
                lines.append("Page excerpt:")
                lines.append(src["page_excerpt"])
            blocks.append("\n".join(lines))

        # Rough token estimate (≈4 chars/token). Trim from the end
        # (lowest-ranked sources) until within budget. We track the running
        # character total instead of re-joining every block on each pop, which
        # would be O(n²) on large contexts.
        char_budget = max_tokens * 4
        # Total length of "\n\n".join(blocks): sum of block lengths plus separators.
        total_chars = sum(len(b) for b in blocks) + max(0, len(blocks) - 1) * 2
        while blocks and total_chars > char_budget:
            removed = blocks.pop()
            total_chars -= len(removed) + (2 if blocks else 0)

        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Citation validation (post-processing, no LLM call)
# ---------------------------------------------------------------------------

def validate_citations(response_text: str, source_registry: dict[int, dict]) -> str:
    """Validate and clean up citations in the LLM's response.

    1. Strip phantom citations (referencing non-existent source IDs).
    """
    body = response_text
    cited_ids = set(int(m) for m in re.findall(r"\[(\d+)\]", body))

    # Check for phantom citations
    valid_ids = set(source_registry.keys())
    phantom_ids = cited_ids - valid_ids
    if phantom_ids:
        for pid in phantom_ids:
            body = body.replace(f"[{pid}]", "")
            print(f"  ⚠ Stripped phantom citation [{pid}]", file=sys.stderr)
        cited_ids -= phantom_ids

    return body.rstrip()


def cited_source_ids(response_text: str, source_registry: dict[int, dict]) -> list[int]:
    """Return valid source IDs cited in the response, in source order."""
    cited_ids = {int(m) for m in re.findall(r"\[(\d+)\]", response_text)}
    return sorted(cited_ids & set(source_registry.keys()))

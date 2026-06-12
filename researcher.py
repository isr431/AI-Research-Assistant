"""Core research pipeline orchestrator."""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from datetime import date
from typing import Any, Callable, Generator

from config import SEARCH_PRESETS, get_brave_api_key, get_provider_config
from fetch_pages import fetch_page_excerpts
from llm import LLMClient
from prompts import (
    PLAN_PROMPT,
    SYNTHESIS_PROMPT_DEEP,
    SYNTHESIS_PROMPT_MODERATE,
    SYNTHESIS_PROMPT_QUICK,
)
from search import brave_image_search, search_parallel
from sources import (
    SourceRegistry,
    cited_source_ids,
    validate_citations,
)
from text_utils import tokenize_terms


def _status(msg: str) -> None:
    """Print a progress indicator to stderr."""
    print(f"🔍 {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Per-stage thinking budget control
# ---------------------------------------------------------------------------
# Not every stage benefits from reasoning equally. Planning is mechanical;
# synthesis is where deeper reasoning pays off.
#
#   Plan:          0% of budget  — simple query decomposition, no reasoning
#   Synthesis:   100% of budget  — full reasoning for the final answer

_STAGE_THINKING_FRACTION = {
    "plan": 0.0,
    "synthesis": 1.0,
}


@contextmanager
def _stage_thinking(
    llm: LLMClient, stage: str, base_budget: int, preset: dict[str, Any] | None = None
) -> Generator[None, None, None]:
    """Temporarily set the LLM thinking budget for a specific pipeline stage."""
    budget_key = f"{stage}_budget"
    if preset and budget_key in preset:
        stage_budget = preset[budget_key]
    else:
        fraction = _STAGE_THINKING_FRACTION.get(stage, 1.0)
        stage_budget = int(base_budget * fraction)
    prev = llm.thinking_budget
    llm.thinking_budget = stage_budget
    try:
        yield
    finally:
        llm.thinking_budget = prev

# ---------------------------------------------------------------------------
# Streaming pipeline events
# ---------------------------------------------------------------------------

# Type alias for the event callback
EventCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class SearchCancelled(Exception):
    """Raised when a running search is cancelled by the user."""


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check and cancel_check():
        raise SearchCancelled("Search cancelled.")


def _source_list(
    registry: SourceRegistry,
    source_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build source summaries for UI/history payloads."""
    sources = registry.all()
    ids = source_ids if source_ids is not None else sorted(sources)
    source_list = []
    for sid in ids:
        src = sources.get(sid)
        if not src:
            continue
        source_list.append({
            "id": sid,
            "title": src["title"],
            "url": src["url"],
            "domain": src["domain"],
            "date": src.get("date", ""),
            "query_origins": src.get("query_origins", []),
            "page_fetch_status": src.get("page_fetch_status", ""),
            "has_page_excerpt": bool(src.get("page_excerpt")),
        })
    return source_list


def _pipeline_mode(preset_name: str, preset: dict[str, Any]) -> str:
    if preset_name == "quick":
        return "quick"
    if preset.get("output_style") == "report":
        return "deep"
    return "moderate"


def _emit_query_event(
    emit: EventCallback,
    queries: list[str],
    *,
    phase: str,
    pass_num: int,
    label: str,
) -> None:
    emit({
        "type": "queries",
        "queries": queries,
        "phase": phase,
        "pass": pass_num,
        "label": label,
    })


def research_stream(
    question: str,
    preset_name: str = "moderate",
    provider_name: str | None = None,
    thinking_enabled: bool = True,
    include_images: bool = True,
    on_event: EventCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> str:
    """Run a research request and emit structured progress events.

    Quick mode uses one direct search pass. Moderate and deep modes add query
    planning plus deterministic coverage checks before final synthesis.
    Returns the final validated response text.
    """
    def emit(event: dict[str, Any]) -> None:
        if on_event:
            on_event(event)

    preset = SEARCH_PRESETS[preset_name]
    provider_cfg = get_provider_config(provider_name)
    brave_key = get_brave_api_key()
    image_executor: ThreadPoolExecutor | None = None
    image_future: Future[list[dict[str, Any]]] | None = None
    image_results: list[dict[str, Any]] = []
    image_results_checked = False

    if include_images:
        image_executor = ThreadPoolExecutor(max_workers=1)
        image_future = image_executor.submit(
            brave_image_search,
            question,
            brave_key,
            count=4,
        )

    # Thinking budget: on = preset default, off = 0
    base_budget = preset["thinking_budget"] if thinking_enabled else 0
    llm = LLMClient(provider_cfg, thinking_budget=base_budget, verbose=False)
    today = date.today().isoformat()

    emit({
        "type": "status",
        "message": f"Using {provider_cfg['name']} ({preset_name} mode)",
        "stage": "init",
    })
    _raise_if_cancelled(cancel_check)

    try:
        # --- Plan queries --------------------------------------------------
        pipeline_mode = _pipeline_mode(preset_name, preset)
        if pipeline_mode == "quick":
            emit({
                "type": "status",
                "message": "Preparing query…",
                "stage": "plan",
            })
            queries = [question]
            _emit_query_event(
                emit,
                queries,
                phase="initial",
                pass_num=1,
                label="Direct search query",
            )
        else:
            emit({
                "type": "status",
                "message": "Planning focused research queries…",
                "stage": "plan",
            })
            try:
                with _stage_thinking(llm, "plan", base_budget, preset):
                    queries = _stage_plan(llm, question, preset, today)
            except Exception as exc:
                emit({
                    "type": "status",
                    "message": (
                        f"Planning failed ({exc}), falling back to direct search."
                    ),
                    "stage": "plan",
                })
                queries = [question]
            _emit_query_event(
                emit,
                queries,
                phase="initial",
                pass_num=1,
                label="Initial research queries",
            )

        _raise_if_cancelled(cancel_check)

        # --- Search and read top sources ----------------------------------
        emit({
            "type": "status",
            "message": "Searching web…",
            "stage": "search",
        })
        registry = SourceRegistry()
        _stage_search_and_harvest(
            queries,
            brave_key,
            preset,
            registry,
            question,
            today,
            emit,
            cancel_check,
            pass_num=1,
            phase="initial",
        )
        _raise_if_cancelled(cancel_check)

        emit({
            "type": "status",
            "message": f"Collected {len(registry)} unique sources",
            "stage": "search",
        })

        if len(registry) == 0:
            no_results = (
                "I wasn't able to find any relevant information for your question. "
                "Please try rephrasing or broadening your query."
            )
            emit({"type": "content", "delta": no_results})
            image_results, image_results_checked = _maybe_emit_images(
                image_future, emit, image_results_checked, timeout=0.1
            )
            emit({
                "type": "done",
                "content": no_results,
                "thinking": "",
                "sources": [],
                "images": image_results,
            })
            return no_results

        # Emit sources and any image results already available.
        emit({"type": "sources", "sources": _source_list(registry)})
        image_results, image_results_checked = _maybe_emit_images(
            image_future, emit, image_results_checked, timeout=0.01
        )

        # --- Synthesize with streaming ------------------------------------
        if pipeline_mode == "quick":
            response, thinking = _quick_pipeline_stream(
                llm, question, preset, registry, today,
                base_budget, emit, cancel_check,
            )
        else:
            response, thinking = _research_pipeline_stream(
                llm, question, queries, brave_key, preset, registry, today,
                base_budget, emit, cancel_check,
            )
        _raise_if_cancelled(cancel_check)

        # --- Validate citations -------------------------------------------
        emit({
            "type": "status",
            "message": "Validating citations…",
            "stage": "synthesis",
        })
        response = validate_citations(response, registry.all())

        # Re-collect sources after validation, keeping only cited sources.
        final_sources = _source_list(
            registry,
            cited_source_ids(response, registry.all()),
        )

        if not image_results_checked:
            image_results, image_results_checked = _maybe_emit_images(
                image_future, emit, image_results_checked, timeout=0.5
            )

        emit({
            "type": "done",
            "content": response,
            "thinking": thinking,
            "sources": final_sources,
            "images": image_results,
        })

        return response
    finally:
        if image_executor:
            image_executor.shutdown(wait=False, cancel_futures=True)


def _maybe_emit_images(
    image_future: Future[list[dict[str, Any]]] | None,
    emit: EventCallback,
    already_checked: bool,
    *,
    timeout: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Emit image results if the optional background image search is ready."""
    if not image_future or already_checked:
        return [], already_checked
    try:
        images = image_future.result(timeout=timeout)
    except TimeoutError:
        return [], False
    except Exception as exc:
        print(f"  ⚠ Image search failed unexpectedly: {exc}", file=sys.stderr)
        return [], True
    if images:
        emit({"type": "images", "images": images})
    return images, True


# ---------------------------------------------------------------------------
# Stage 1: Plan
# ---------------------------------------------------------------------------

def _stage_plan(
    llm: LLMClient,
    question: str,
    preset: dict[str, Any],
    current_date: str,
) -> list[str]:
    """Generate sub-queries via the merged plan prompt."""
    _status("Planning search…")
    n = preset["sub_queries"]
    prompt = PLAN_PROMPT.format(question=question, n=n, current_date=current_date)
    result = llm.ask_json(prompt)

    queries = _coerce_query_list(result.get("queries", []), n)
    if not queries:
        raise ValueError("LLM returned no queries")

    _status(f"Generated {len(queries)} sub-queries")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}", file=sys.stderr)
    return queries


def _coerce_query_list(raw_queries: Any, limit: int) -> list[str]:
    """Return a bounded list of non-empty query strings from model JSON."""
    if limit <= 0:
        return []
    if not isinstance(raw_queries, list):
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in raw_queries:
        query = str(raw_query).strip()
        if not query or query in seen:
            continue
        queries.append(query)
        seen.add(query)
        if len(queries) >= limit:
            break
    return queries


# ---------------------------------------------------------------------------
# Search and source enrichment
# ---------------------------------------------------------------------------

def _stage_search_and_harvest(
    queries: list[str],
    brave_key: str,
    preset: dict[str, Any],
    registry: SourceRegistry,
    question: str,
    current_date: str,
    emit: EventCallback | None = None,
    cancel_check: CancelCheck | None = None,
    pass_num: int | None = None,
    phase: str = "initial",
) -> None:
    """Search in parallel and register all results in the source registry."""
    _raise_if_cancelled(cancel_check)
    _status(f"Searching [{len(queries)} queries]…")
    results = search_parallel(
        queries,
        brave_key,
        max_urls=preset["urls_per_query"],
        max_tokens=preset["tokens_per_query"],
        should_cancel=cancel_check,
    )
    _raise_if_cancelled(cancel_check)

    for r in results:
        registry.add(r)

    registry.score_sources(question, queries, current_date)
    _stage_fetch_top_pages(
        queries,
        preset,
        registry,
        question,
        emit,
        cancel_check,
        pass_num=pass_num,
        phase=phase,
    )
    _status(f"Collected {len(registry)} unique sources")


def _stage_fetch_top_pages(
    queries: list[str],
    preset: dict[str, Any],
    registry: SourceRegistry,
    question: str,
    emit: EventCallback | None = None,
    cancel_check: CancelCheck | None = None,
    pass_num: int | None = None,
    phase: str = "initial",
) -> None:
    """Fetch readable excerpts for the highest-scored sources."""
    _raise_if_cancelled(cancel_check)
    limit = int(preset.get("full_page_sources", 0) or 0)
    if limit <= 0:
        return
    total_limit = int(preset.get("total_full_page_sources", limit) or limit)
    already_attempted = sum(
        1
        for src in registry.all().values()
        if src.get("page_excerpt") or src.get("page_fetch_status")
    )
    remaining = max(0, total_limit - already_attempted)
    limit = min(limit, remaining)
    if limit <= 0:
        if emit:
            emit({
                "type": "source_fetch",
                "mode": "top_ranked",
                "pass": pass_num,
                "phase": phase,
                "sources": [],
                "summary": f"Fetch budget: {already_attempted}/{total_limit}",
            })
        return

    candidates = registry.top_sources_for_fetch(limit)
    if not candidates:
        return

    _status(f"Fetching full text for {len(candidates)} top sources…")
    if emit:
        emit({
            "type": "status",
            "message": "Reading sources…",
            "stage": "search",
        })
    query_text = " ".join([question, *queries])
    fetched = fetch_page_excerpts(
        candidates,
        query_text,
        max_chars=int(preset.get("full_page_chars", 6_000)),
        should_cancel=cancel_check,
    )
    _raise_if_cancelled(cancel_check)
    for sid, result in fetched.items():
        registry.set_page_excerpt(
            sid,
            result.get("excerpt", ""),
            result.get("status", "unknown"),
        )
    if emit:
        fetch_events: list[dict[str, Any]] = []
        for source in candidates:
            sid = int(source["id"])
            updated = registry.get(sid) or {}
            fetch_events.append({
                "id": sid,
                "title": updated.get("title", source.get("title", "")),
                "url": updated.get("url", source.get("url", "")),
                "domain": updated.get("domain", source.get("domain", "")),
                "page_fetch_status": updated.get("page_fetch_status", ""),
                "has_page_excerpt": bool(updated.get("page_excerpt")),
            })
        emit({
            "type": "source_fetch",
            "mode": "top_ranked",
            "pass": pass_num,
            "phase": phase,
            "sources": fetch_events,
            "summary": f"Sources read: {len(fetch_events)}",
        })


# ---------------------------------------------------------------------------
# Deterministic coverage checks for research modes
# ---------------------------------------------------------------------------


def _coverage_terms(text: str) -> set[str]:
    return tokenize_terms(text)


def _source_coverage_text(src: dict[str, Any]) -> str:
    return " ".join([
        str(src.get("title", "")),
        str(src.get("domain", "")),
        " ".join(str(snippet) for snippet in src.get("snippets", [])),
        str(src.get("page_excerpt", "")),
    ])


def _compact_query(text: str, max_terms: int = 12) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", text):
        term = raw.strip("-")
        key = term.lower()
        if key in seen:
            continue
        terms.append(term)
        seen.add(key)
        if len(terms) >= max_terms:
            break
    return " ".join(terms).strip() or text.strip()


def _build_followup_query(question: str, row: dict[str, Any]) -> str:
    missing_terms = row.get("missing_terms", [])
    missing_text = " ".join(str(term) for term in missing_terms[:4])
    return _compact_query(
        " ".join([
            str(row.get("question", "")),
            missing_text,
            question,
        ])
    )


def _deterministic_coverage_analysis(
    question: str,
    queries: list[str],
    registry: SourceRegistry,
    *,
    max_followups: int,
    previous_queries: list[str],
) -> dict[str, Any]:
    """Estimate query coverage from collected source text without an LLM call."""
    rows: list[dict[str, Any]] = []
    sources = registry.all()

    for query in queries:
        terms = _coverage_terms(query) or _coverage_terms(question)
        best_score = 0.0
        best_terms: set[str] = set()
        matched_sources: list[int] = []

        for sid, src in sources.items():
            source_terms = _coverage_terms(_source_coverage_text(src))
            matched = terms & source_terms
            origin_match = query in src.get("query_origins", [])
            score = len(matched) / max(1, len(terms))
            if origin_match:
                score += 0.20
            if matched or origin_match:
                matched_sources.append(sid)
            if score > best_score:
                best_score = score
                best_terms = matched

        matched_sources = sorted(set(matched_sources))[:4]
        missing_terms = sorted(terms - best_terms)[:6]
        if best_score >= 0.34 or len(best_terms) >= 3:
            status = "answered"
            detail = (
                f"Evidence found in {len(matched_sources)} "
                f"{'source' if len(matched_sources) == 1 else 'sources'}."
            )
        elif best_score >= 0.18 or matched_sources:
            status = "partial"
            detail = "Some evidence found, but important query terms are thin."
        else:
            status = "unanswered"
            detail = "No strong matching evidence found in collected sources."

        rows.append({
            "question": query,
            "status": status,
            "key_findings": detail if status == "answered" else "",
            "missing": detail if status != "answered" else "",
            "score": round(best_score, 3),
            "matched_sources": matched_sources,
            "matched_terms": sorted(best_terms)[:8],
            "missing_terms": missing_terms,
        })

    previous = {query.strip().lower() for query in previous_queries}
    followups: list[str] = []
    weak_rows = sorted(
        [row for row in rows if row["status"] != "answered"],
        key=lambda row: (
            0 if row["status"] == "unanswered" else 1,
            float(row.get("score", 0.0)),
        ),
    )
    for row in weak_rows:
        followup = _build_followup_query(question, row)
        key = followup.lower()
        if not followup or key in previous or key in {q.lower() for q in followups}:
            continue
        followups.append(followup)
        if len(followups) >= max_followups:
            break

    return {
        "strategy": "deterministic token-overlap coverage",
        "summary": f"Queries checked: {len(rows)}",
        "answered": rows,
        "followup_queries": followups,
    }


# ---------------------------------------------------------------------------
# Quick pipeline
# ---------------------------------------------------------------------------

def _quick_pipeline_stream(
    llm: LLMClient,
    question: str,
    preset: dict[str, Any],
    registry: SourceRegistry,
    current_date: str,
    base_budget: int,
    emit: EventCallback,
    cancel_check: CancelCheck | None = None,
) -> tuple[str, str]:
    """Quick mode: one retrieval pass and one streaming synthesis."""
    emit({
        "type": "status",
        "message": "Checking collected sources…",
        "stage": "gap_analysis",
    })
    emit({
        "type": "gap_analysis",
        "mode": "quick",
        "pass": 1,
        "result": {
            "strategy": "single-pass source check",
            "summary": f"Sources collected: {len(registry)}",
            "answered": [
                {
                    "question": question,
                    "status": "answered" if len(registry) else "unanswered",
                    "key_findings": f"{len(registry)} unique sources collected.",
                    "score": None,
                    "matched_sources": sorted(registry.all())[:4],
                    "matched_terms": [],
                    "missing_terms": [],
                }
            ],
            "followup_queries": [],
        },
    })
    context = registry.format_knowledge_context(
        max_tokens=preset["max_context_tokens"]
    )

    emit({"type": "status", "message": "Writing response…", "stage": "synthesis"})
    prompt = SYNTHESIS_PROMPT_QUICK.format(
        question=question, knowledge_context=context, current_date=current_date
    )

    response = ""
    thinking = ""
    with _stage_thinking(llm, "synthesis", base_budget, preset):
        for event in llm.ask_text_stream(prompt):
            _raise_if_cancelled(cancel_check)
            if event["type"] == "done":
                response = event["content"]
                thinking = event["thinking"]
            else:
                emit(event)

    return response, thinking


# ---------------------------------------------------------------------------
# Moderate/deep research pipeline
# ---------------------------------------------------------------------------

def _research_pipeline_stream(
    llm: LLMClient,
    question: str,
    queries: list[str],
    brave_key: str,
    preset: dict[str, Any],
    registry: SourceRegistry,
    current_date: str,
    base_budget: int,
    emit: EventCallback,
    cancel_check: CancelCheck | None = None,
) -> tuple[str, str]:
    """Research mode with deterministic coverage passes and one final synthesis."""
    max_passes = preset["max_passes"]
    mode = "deep" if preset.get("output_style") == "report" else "moderate"
    all_queries = list(queries)

    for pass_num in range(1, max_passes):
        _raise_if_cancelled(cancel_check)

        emit({
            "type": "status",
            "message": "Checking collected sources…",
            "stage": "gap_analysis",
        })
        coverage = _deterministic_coverage_analysis(
            question,
            all_queries,
            registry,
            max_followups=int(preset.get("followup_queries_per_pass", 2) or 0),
            previous_queries=all_queries,
        )
        _raise_if_cancelled(cancel_check)
        emit({
            "type": "gap_analysis",
            "mode": mode,
            "pass": pass_num,
            "result": coverage,
        })

        followup = _coerce_query_list(
            coverage.get("followup_queries", []),
            int(preset.get("followup_queries_per_pass", 2) or 0),
        )
        if not followup:
            emit({
                "type": "status",
                "message": "Coverage check complete.",
                "stage": "gap_analysis",
            })
            break

        emit({
            "type": "status",
            "message": (
                "Searching follow-up queries…"
            ),
            "stage": "search",
        })
        _emit_query_event(
            emit,
            followup,
            phase="followup",
            pass_num=pass_num + 1,
            label=f"Follow-up queries from coverage pass {pass_num}",
        )

        _stage_search_and_harvest(
            followup,
            brave_key,
            preset,
            registry,
            question,
            current_date,
            emit,
            cancel_check,
            pass_num=pass_num + 1,
            phase="followup",
        )

        # Emit updated source list
        emit({"type": "sources", "sources": _source_list(registry)})

        all_queries = all_queries + followup
    else:
        emit({
            "type": "status",
            "message": "Coverage check complete.",
            "stage": "gap_analysis",
        })

    # Final synthesis with streaming
    context = registry.format_knowledge_context(
        max_tokens=preset["max_context_tokens"]
    )
    emit({
        "type": "status",
        "message": "Writing comprehensive report…",
        "stage": "synthesis",
    })
    prompt_template = SYNTHESIS_PROMPT_DEEP if preset.get("output_style") == "report" else SYNTHESIS_PROMPT_MODERATE
    prompt = prompt_template.format(
        question=question, knowledge_context=context, current_date=current_date
    )

    response = ""
    thinking = ""
    with _stage_thinking(llm, "synthesis", base_budget, preset):
        for event in llm.ask_text_stream(prompt):
            _raise_if_cancelled(cancel_check)
            if event["type"] == "done":
                response = event["content"]
                thinking = event["thinking"]
            else:
                emit(event)

    return response, thinking

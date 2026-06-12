"""Brave Search LLM Context API client with parallel execution."""

from __future__ import annotations

import sys
import time
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable
from urllib.parse import urlparse

import requests


# ---------------------------------------------------------------------------
# Single-query search
# ---------------------------------------------------------------------------

def brave_llm_search(
    query: str,
    api_key: str,
    *,
    max_urls: int = 5,
    max_tokens: int = 4096,
) -> list[dict[str, Any]]:
    """Call the Brave LLM Context endpoint for a single query.

    Returns a list of result dicts:
    ``{url, title, domain, snippets, date, query_origin}``

    Retries once on transient errors (429, 5xx) with a 2 s backoff.
    Returns an empty list (with a stderr warning) if no results or on failure.
    """
    url = "https://api.search.brave.com/res/v1/llm/context"
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "maximum_number_of_urls": max_urls,
        "maximum_number_of_tokens": max_tokens,
    }

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries:
                    wait = 2 * attempt
                    print(
                        f"  ⚠ Brave API returned {resp.status_code} for "
                        f"query '{query[:40]}…', retrying in {wait}s…",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                print(
                    f"  ⚠ Brave API error {resp.status_code} for "
                    f"query '{query[:40]}…' — skipping.",
                    file=sys.stderr,
                )
                return []

            resp.raise_for_status()
            data = resp.json()

            # Extract results from grounding.generic[]
            generic = (
                data.get("grounding", {}).get("generic")
                or data.get("results", [])
            )
            if not generic:
                print(
                    f"  ⚠ No results for query: '{query[:50]}…'",
                    file=sys.stderr,
                )
                return []

            results: list[dict[str, Any]] = []
            for rank, item in enumerate(generic, 1):
                results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", "Untitled"),
                    "domain": item.get("meta", {}).get("hostname", "")
                    or _extract_domain(item.get("url", "")),
                    "snippets": item.get("snippets", []),
                    "date": item.get("meta", {}).get("date", ""),
                    "query_origin": query,
                    "result_rank": rank,
                })
            return results

        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(
                f"  ⚠ Search failed for '{query[:40]}…': {exc}",
                file=sys.stderr,
            )
            return []

    return []  # should not reach here


# ---------------------------------------------------------------------------
# Image search
# ---------------------------------------------------------------------------

def brave_image_search(
    query: str,
    api_key: str,
    *,
    count: int | None = None,
) -> list[dict[str, Any]]:
    """Call the Brave Image Search API and return normalized image results."""
    result_count = 4 if count is None else max(0, count)
    if result_count <= 0:
        return []

    url = "https://api.search.brave.com/res/v1/images/search"
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    requested_count = min(result_count, 4)
    fetch_count = min(max(requested_count * 3, requested_count), 12)
    params = {
        "q": query,
        "count": fetch_count,
        "safesearch": "off",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(
            f"  ⚠ Image search failed for '{query[:40]}…': {exc}",
            file=sys.stderr,
        )
        return []

    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        return []

    images: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, item in enumerate(raw_results, 1):
        normalized = _normalize_image_result(item, rank)
        image_key = normalized.get("image_url") or normalized.get("thumbnail_url")
        if not image_key or image_key in seen:
            continue
        seen.add(image_key)
        images.append(normalized)
        if len(images) >= fetch_count:
            break
    ranked_images = sorted(
        images,
        key=lambda image: (
            -float(image.get("quality_score", 0.0)),
            int(image.get("rank") or 999),
        ),
    )
    return _select_unique_images(ranked_images, requested_count)


def _normalize_image_result(item: dict[str, Any], rank: int) -> dict[str, Any]:
    thumbnail = item.get("thumbnail") if isinstance(item.get("thumbnail"), dict) else {}
    properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
    meta_url = item.get("meta_url") if isinstance(item.get("meta_url"), dict) else {}
    page_url = item.get("url") or item.get("source") or meta_url.get("netloc") or ""
    image_url = item.get("image_url") or item.get("src") or properties.get("url") or ""
    thumbnail_url = item.get("thumbnail_url") or thumbnail.get("src") or thumbnail.get("url") or ""
    source_domain = (
        item.get("source_domain")
        or item.get("source")
        or meta_url.get("hostname")
        or _extract_domain(page_url)
    )
    image = {
        "title": item.get("title") or item.get("description") or "Image result",
        "url": page_url,
        "thumbnail_url": thumbnail_url,
        "image_url": image_url,
        "source_domain": source_domain,
        "width": properties.get("width") or item.get("width"),
        "height": properties.get("height") or item.get("height"),
        "description": item.get("description", ""),
        "rank": rank,
    }
    image["quality_score"] = _image_quality_score(image)
    return image


def _image_quality_score(image: dict[str, Any]) -> float:
    width = _safe_int(image.get("width"))
    height = _safe_int(image.get("height"))
    area = width * height
    title = str(image.get("title") or "")
    description = str(image.get("description") or "")

    score = 0.0
    if image.get("thumbnail_url"):
        score += 1.0
    if image.get("image_url"):
        score += 1.0
    if image.get("url"):
        score += 0.5
    if image.get("source_domain"):
        score += 0.3
    if title and title != "Image result":
        score += 0.4
    if description:
        score += 0.2

    if area >= 1_000_000:
        score += 3.0
    elif area >= 480_000:
        score += 2.4
    elif area >= 250_000:
        score += 1.8
    elif area >= 100_000:
        score += 1.0
    elif area > 0:
        score += 0.2

    if width and height:
        ratio = max(width / height, height / width)
        if ratio <= 2.2:
            score += 0.6
        elif ratio >= 4.0:
            score -= 0.8

    rank = _safe_int(image.get("rank")) or 99
    score += max(0.0, 1.0 - ((rank - 1) * 0.08))
    return round(score, 4)


def _select_unique_images(
    images: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for image in images:
        if any(_images_look_duplicate(image, existing) for existing in selected):
            continue
        selected.append(image)
        if len(selected) >= limit:
            break
    return selected


def _images_look_duplicate(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_urls = _image_identity_urls(a)
    b_urls = _image_identity_urls(b)
    if a_urls & b_urls:
        return True

    a_filenames = _image_filename_signatures(a)
    b_filenames = _image_filename_signatures(b)
    if a_filenames & b_filenames:
        return True
    return False


def _image_identity_urls(image: dict[str, Any]) -> set[str]:
    urls = set()
    for key in ("image_url", "thumbnail_url"):
        normalized = _normalize_image_url(str(image.get(key) or ""))
        if normalized:
            urls.add(normalized)
    return urls


def _normalize_image_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if not parsed.netloc:
        return ""
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    return f"{parsed.netloc.lower().removeprefix('www.')}{path.lower()}"


def _image_filename_signatures(image: dict[str, Any]) -> set[str]:
    signatures = set()
    for key in ("image_url", "thumbnail_url"):
        filename = _image_filename_signature(str(image.get(key) or ""))
        if filename:
            signatures.add(filename)
    return signatures


def _image_filename_signature(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    filename = parsed.path.rsplit("/", 1)[-1].lower()
    filename = re.sub(r"\.(?:avif|gif|jpe?g|png|webp)$", "", filename)
    filename = re.sub(r"[_-](?:\d{2,5}x\d{2,5}|w\d{2,5}|h\d{2,5})$", "", filename)
    filename = re.sub(r"[^a-z0-9]+", "-", filename).strip("-")
    if len(filename) < 10 or filename in {"image", "photo", "picture", "thumbnail"}:
        return ""
    return filename


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Parallel search
# ---------------------------------------------------------------------------

def search_parallel(
    queries: list[str],
    api_key: str,
    *,
    max_urls: int = 5,
    max_tokens: int = 4096,
    max_workers: int = 3,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Execute multiple search queries concurrently.

    Returns a flat list of all result dicts from all queries.
    Queries that return 0 results are silently skipped (warning already
    logged by ``brave_llm_search``).
    """
    all_results: list[dict[str, Any]] = []
    if should_cancel and should_cancel():
        return all_results

    cancelled = False
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_query = {
            pool.submit(
                brave_llm_search, q, api_key,
                max_urls=max_urls, max_tokens=max_tokens,
            ): (idx, q)
            for idx, q in enumerate(queries)
            if not (should_cancel and should_cancel())
        }

        pending = set(future_to_query)
        while pending:
            if should_cancel and should_cancel():
                cancelled = True
                for future in pending:
                    future.cancel()
                break

            done, pending = wait(
                pending,
                timeout=0.2,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue

            for future in done:
                query_index, q = future_to_query[future]
                if future.cancelled():
                    continue
                if should_cancel and should_cancel():
                    continue
                try:
                    results = future.result()
                    for result in results:
                        result["query_index"] = query_index
                    all_results.extend(results)
                except Exception as exc:
                    print(
                        f"  ⚠ Unexpected error for '{q[:40]}…': {exc}",
                        file=sys.stderr,
                    )

        if should_cancel and should_cancel():
            cancelled = True
            for future in pending:
                future.cancel()
    finally:
        pool.shutdown(wait=not cancelled, cancel_futures=cancelled)

    return sorted(
        all_results,
        key=lambda r: (r.get("query_index", 0), r.get("result_rank", 0)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Best-effort domain extraction without urllib."""
    try:
        hostname = urlparse(url).hostname
        if hostname:
            return hostname.removeprefix("www.")
    except ValueError:
        pass
    url = url.split("//", 1)[-1]
    return url.split("/", 1)[0].removeprefix("www.")

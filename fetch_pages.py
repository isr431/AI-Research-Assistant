"""Fetch and extract readable excerpts from top-ranked source pages."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Callable

import requests

from text_utils import tokenize_terms

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AIResearchAssistant/1.0; "
        "+https://localhost)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "form",
    "nav",
    "footer",
}

_BLOCKED_PAGE_MARKERS = (
    "awswaf",
    "gokuprops",
    "captcha",
    "verify you are human",
    "access denied",
    "unusual traffic",
)


class _ReadableTextParser(HTMLParser):
    """Best-effort readable text extraction using the standard library."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag.lower() in {"p", "br", "li", "div", "section", "article", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag.lower() in {"p", "li", "div", "section", "article", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        raw = " ".join(self._parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*", "\n", raw)
        return raw.strip()


def _tokenize(text: str) -> set[str]:
    return tokenize_terms(text)


def _read_limited(resp: requests.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            keep = len(chunk) - (total - max_bytes)
            if keep > 0:
                chunks.append(chunk[:keep])
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_text_with_trafilatura(html: str, url: str) -> str:
    """Use trafilatura when installed; return empty string on fallback."""
    try:
        import trafilatura  # type: ignore[import-not-found]
    except ImportError:
        return ""

    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            output_format="txt",
        )
    except Exception:
        return ""

    return re.sub(r"\n{3,}", "\n\n", extracted or "").strip()


def _extract_text(html: str, url: str = "") -> str:
    richer = _extract_text_with_trafilatura(html, url)
    if len(richer) >= 200:
        return richer

    parser = _ReadableTextParser()
    parser.feed(html)
    parser.close()
    return parser.text()


def _blocked_page_status(html: str, status_code: int = 0) -> str:
    """Return a fetch status when the response is an anti-bot/block page."""
    lowered = html[:20_000].lower()
    if any(marker in lowered for marker in _BLOCKED_PAGE_MARKERS):
        return "blocked by site anti-bot challenge"
    if status_code in {401, 403, 429}:
        return f"blocked by site: HTTP {status_code}"
    return ""


def _best_excerpt(text: str, query_text: str, max_chars: int) -> str:
    terms = _tokenize(query_text)
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z0-9])", text)
    ]
    paragraphs = [p for p in paragraphs if len(p) >= 80]
    if not paragraphs:
        return text[:max_chars].strip()

    scored: list[tuple[float, int, str]] = []
    for idx, paragraph in enumerate(paragraphs):
        words = _tokenize(paragraph)
        overlap = len(words & terms)
        score = overlap + min(len(paragraph), 1200) / 2400
        scored.append((score, idx, paragraph))

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:8]
    selected = sorted(selected, key=lambda item: item[1])

    excerpt_parts: list[str] = []
    total = 0
    for _, _, paragraph in selected:
        addition = len(paragraph) + (2 if excerpt_parts else 0)
        if total + addition > max_chars:
            remaining = max_chars - total - (2 if excerpt_parts else 0)
            if remaining > 120:
                excerpt_parts.append(paragraph[:remaining].rsplit(" ", 1)[0])
            break
        excerpt_parts.append(paragraph)
        total += addition

    return "\n\n".join(excerpt_parts).strip()


def fetch_page_excerpt(
    url: str,
    query_text: str,
    *,
    timeout: int = 5,
    max_bytes: int = 600_000,
    max_chars: int = 6_000,
) -> tuple[str, str]:
    """Fetch one URL and return ``(excerpt, status)``.

    The caller should treat failures as non-fatal. ``status`` is a short reason
    suitable for debugging or source metadata.
    """
    try:
        with requests.get(
            url,
            headers=_HEADERS,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        ) as resp:
            status_code = resp.status_code
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if "html" not in content_type and "text/plain" not in content_type:
                return "", f"unsupported content type: {content_type or 'unknown'}"

            raw = _read_limited(resp, max_bytes)
            encoding = resp.encoding or resp.apparent_encoding or "utf-8"
    except requests.RequestException as exc:
        return "", f"fetch failed: {exc}"

    try:
        decoded = raw.decode(encoding, errors="replace")
        blocked_status = _blocked_page_status(decoded, status_code)
        if blocked_status:
            return "", blocked_status
        text = _extract_text(decoded, url)
    except Exception as exc:
        return "", f"extract failed: {exc}"

    if len(text) < 200:
        return "", "page text too short"

    return _best_excerpt(text, query_text, max_chars), "fetched"


def fetch_page_excerpts(
    sources: list[dict[str, Any]],
    query_text: str,
    *,
    max_workers: int = 3,
    max_chars: int = 6_000,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[int, dict[str, str]]:
    """Fetch excerpts for source summaries with ``id`` and ``url`` fields."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    fetched: dict[int, dict[str, str]] = {}
    if should_cancel and should_cancel():
        return fetched

    pool = ThreadPoolExecutor(max_workers=max_workers)
    future_to_source = {
        pool.submit(
            fetch_page_excerpt,
            source["url"],
            query_text,
            max_chars=max_chars,
        ): source
        for source in sources
        if source.get("id")
        and source.get("url")
        and not (should_cancel and should_cancel())
    }
    try:
        pending = set(future_to_source)
        while pending:
            if should_cancel and should_cancel():
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
                source = future_to_source[future]
                if future.cancelled():
                    continue
                if should_cancel and should_cancel():
                    continue
                try:
                    excerpt, status = future.result()
                except Exception as exc:
                    excerpt, status = "", f"fetch failed: {exc}"
                fetched[int(source["id"])] = {
                    "excerpt": excerpt,
                    "status": status,
                }
    finally:
        cancelled = bool(should_cancel and should_cancel())
        for future in future_to_source:
            if cancelled:
                future.cancel()
        pool.shutdown(wait=not cancelled, cancel_futures=cancelled)
    return fetched

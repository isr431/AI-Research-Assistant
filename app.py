"""Flask web server for the AI Research Assistant GUI.

Provides SSE-based streaming of research pipeline events to the browser.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from queue import Empty, Queue
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory

from config import MODEL_PROVIDERS, DEFAULT_PROVIDER, normalize_provider_name
from history import SearchHistory
from researcher import SearchCancelled, research_stream

app = Flask(__name__, static_folder="static", static_url_path="/static")

# In-memory store for active searches
_active_searches: dict[str, dict[str, Any]] = {}
_search_lock = threading.Lock()
_COMPLETED_SEARCH_TTL_SECONDS = 600

# Shared history manager
_history = SearchHistory(output_dir="output")


def _cleanup_active_searches() -> None:
    """Remove completed searches that no client consumed after a grace period."""
    cutoff = time.monotonic() - _COMPLETED_SEARCH_TTL_SECONDS
    with _search_lock:
        stale_ids = [
            search_id
            for search_id, search in _active_searches.items()
            if search.get("done") and float(search.get("completed_at", 0.0)) < cutoff
        ]
        for search_id in stale_ids:
            _active_searches.pop(search_id, None)


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# API: Providers
# ---------------------------------------------------------------------------

@app.route("/api/providers")
def api_providers():
    providers = {}
    for key, cfg in MODEL_PROVIDERS.items():
        providers[key] = {"name": cfg["name"]}
    return jsonify({"providers": providers, "default": DEFAULT_PROVIDER})


# ---------------------------------------------------------------------------
# API: Start a search
# ---------------------------------------------------------------------------

@app.route("/api/search", methods=["POST"])
def api_search():
    _cleanup_active_searches()
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    provider = normalize_provider_name(data.get("provider", DEFAULT_PROVIDER))
    mode = data.get("mode", "moderate")
    thinking = data.get("thinking", True)
    include_images = data.get("include_images", True)

    if mode not in ("quick", "moderate", "deep"):
        return jsonify({"error": "Mode must be 'quick', 'moderate' or 'deep'"}), 400
    if provider not in MODEL_PROVIDERS:
        return jsonify({"error": f"Unknown provider: {provider}"}), 400

    search_id = uuid.uuid4().hex[:12]
    event_queue: Queue = Queue()
    cancel_event = threading.Event()

    with _search_lock:
        _active_searches[search_id] = {
            "queue": event_queue,
            "question": question,
            "provider": provider,
            "mode": mode,
            "thinking": thinking,
            "include_images": bool(include_images),
            "done": False,
            "created_at": time.monotonic(),
            "completed_at": None,
            "cancel_event": cancel_event,
        }

    # Run research in a background thread
    def run_research():
        final_content = ""
        final_thinking = ""
        final_sources: list[dict] = []
        latest_sources: list[dict] = []
        final_images: list[dict] = []
        gap_analyses: list[dict] = []
        query_events: list[dict] = []
        source_fetch_events: list[dict] = []

        def on_event(event: dict[str, Any]) -> None:
            nonlocal final_content, final_thinking, final_sources, final_images
            nonlocal latest_sources, gap_analyses, source_fetch_events
            if event.get("type") == "sources":
                latest_sources = event.get("sources", [])
            elif event.get("type") == "queries":
                query_events.append({
                    "phase": event.get("phase"),
                    "pass": event.get("pass"),
                    "label": event.get("label"),
                    "queries": event.get("queries", []),
                })
            elif event.get("type") == "images":
                final_images = event.get("images", [])
            elif event.get("type") == "gap_analysis":
                gap_analyses.append({
                    "mode": event.get("mode"),
                    "pass": event.get("pass"),
                    "result": event.get("result", {}),
                })
            elif event.get("type") == "source_fetch":
                source_fetch_events.append({
                    "mode": event.get("mode"),
                    "pass": event.get("pass"),
                    "phase": event.get("phase"),
                    "summary": event.get("summary"),
                    "sources": event.get("sources", []),
                })
            if event.get("type") == "done":
                final_content = event.get("content", "")
                final_thinking = event.get("thinking", "")
                final_sources = event.get("sources", [])
                final_images = event.get("images", final_images)

                # Save to history BEFORE forwarding the done event,
                # so that the client's loadHistory() call finds the entry.
                try:
                    source_fetches = [
                        {
                            "id": src.get("id"),
                            "url": src.get("url"),
                            "title": src.get("title"),
                            "domain": src.get("domain"),
                            "page_fetch_status": src.get("page_fetch_status", ""),
                            "has_page_excerpt": src.get("has_page_excerpt", False),
                        }
                        for src in latest_sources
                    ]
                    _history.save_search(
                        question=question,
                        mode=mode,
                        provider=provider,
                        content=final_content,
                        thinking=final_thinking,
                        sources=final_sources,
                        images=final_images,
                        metadata={
                            "query_events": query_events,
                            "gap_analyses": gap_analyses,
                            "source_fetch_events": source_fetch_events,
                            "source_fetches": source_fetches,
                        },
                    )
                except Exception:
                    pass  # Don't fail the search if saving fails

            event_queue.put(event)

        try:
            research_stream(
                question=question,
                preset_name=mode,
                provider_name=provider,
                thinking_enabled=thinking,
                include_images=bool(include_images),
                on_event=on_event,
                cancel_check=cancel_event.is_set,
            )
        except SearchCancelled:
            event_queue.put({
                "type": "cancelled",
                "message": "Search cancelled.",
            })
        except Exception as exc:
            event_queue.put({
                "type": "error",
                "message": str(exc),
            })

        with _search_lock:
            if search_id in _active_searches:
                _active_searches[search_id]["done"] = True
                _active_searches[search_id]["completed_at"] = time.monotonic()

    thread = threading.Thread(target=run_research, daemon=True)
    thread.start()

    return jsonify({"search_id": search_id})


@app.route("/api/search/<search_id>/cancel", methods=["POST"])
def api_search_cancel(search_id: str):
    _cleanup_active_searches()
    with _search_lock:
        search = _active_searches.get(search_id)
    if not search:
        return jsonify({"error": "Search not found"}), 404

    search["cancel_event"].set()
    search["queue"].put({
        "type": "status",
        "message": "Cancellation requested…",
        "stage": "synthesis",
    })
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API: Stream search events (SSE)
# ---------------------------------------------------------------------------

@app.route("/api/search/<search_id>/stream")
def api_search_stream(search_id: str):
    _cleanup_active_searches()
    with _search_lock:
        search = _active_searches.get(search_id)
    if not search:
        return jsonify({"error": "Search not found"}), 404

    def event_stream():
        queue = search["queue"]
        terminal_event = False
        try:
            while True:
                try:
                    event = queue.get(timeout=300)  # 5 min timeout
                except Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    continue

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                if event.get("type") in ("done", "error", "cancelled"):
                    terminal_event = True
                    break
        finally:
            if not terminal_event:
                search["cancel_event"].set()
            with _search_lock:
                _active_searches.pop(search_id, None)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# API: Search history
# ---------------------------------------------------------------------------

@app.route("/api/history")
def api_history():
    searches = _history.list_searches()
    return jsonify({"searches": searches})


@app.route("/api/history/<search_id>")
def api_history_detail(search_id: str):
    result = _history.get_search(search_id)
    if not result:
        return jsonify({"error": "Search not found"}), 404
    return jsonify(result)


@app.route("/api/history/<search_id>", methods=["DELETE"])
def api_history_delete(search_id: str):
    success = _history.delete_search(search_id)
    if not success:
        return jsonify({"error": "Search not found"}), 404
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=31415, debug=True, threaded=True)

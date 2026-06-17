from __future__ import annotations

import sys
import tempfile
import threading
import unittest
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import app as app_module
import fetch_pages
import researcher
import search
from history import SearchHistory, generate_search_title
from researcher import SearchCancelled, _raise_if_cancelled
from sources import SourceRegistry


class HistoryMetadataTests(unittest.TestCase):
    def test_generate_search_title_is_brief(self) -> None:
        title = generate_search_title(
            "What are the latest Apple WWDC announcements for iOS and macOS?"
        )

        self.assertEqual(title, "The Latest Apple WWDC Announcements For IOS")
        self.assertLessEqual(len(title.split()), 7)

    def test_save_and_load_pipeline_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history = SearchHistory(output_dir=tmpdir)
            metadata = {
                "query_events": [
                    {
                        "phase": "initial",
                        "pass": 1,
                        "label": "Direct search query",
                        "queries": ["Question?"],
                    }
                ],
                "gap_analyses": [
                    {
                        "mode": "quick",
                        "pass": 1,
                        "result": {
                            "strategy": "single-pass quick mode",
                            "summary": "Quick mode uses one direct search pass.",
                            "answered": [],
                            "followup_queries": [],
                        },
                    }
                ],
                "source_fetch_events": [
                    {
                        "mode": "top_ranked",
                        "pass": 1,
                        "phase": "initial",
                        "summary": "Read 1 top-ranked source.",
                        "sources": [
                            {
                                "id": 1,
                                "url": "https://example.com",
                                "page_fetch_status": "fetched",
                                "has_page_excerpt": True,
                            }
                        ],
                    }
                ],
            }

            search_id = history.save_search(
                question="Question?",
                mode="quick",
                provider="test-model",
                content="Answer [1]",
                sources=[{"id": 1, "url": "https://example.com"}],
                images=[{
                    "title": "Example image",
                    "url": "https://example.com/image",
                    "thumbnail_url": "https://example.com/thumb.jpg",
                }],
                metadata=metadata,
            )

            loaded = history.get_search(search_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["content"], "Answer [1]")
            self.assertEqual(loaded["title"], "Question")
            self.assertEqual(len(loaded["images"]), 1)
            self.assertEqual(loaded["images"][0]["title"], "Example image")
            self.assertEqual(loaded["metadata"], metadata)

    def test_concurrent_saves_use_distinct_files_and_index_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history = SearchHistory(output_dir=tmpdir)

            def save_one() -> str:
                return history.save_search(
                    question="Same question?",
                    mode="quick",
                    provider="test-model",
                    content="Answer",
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                ids = list(pool.map(lambda _: save_one(), range(2)))

            entries = history.list_searches()
            filenames = [entry["filename"] for entry in entries]

            self.assertEqual(len(ids), 2)
            self.assertEqual(len(entries), 2)
            self.assertEqual(len(set(filenames)), 2)
            for filename in filenames:
                self.assertTrue(os.path.exists(os.path.join(tmpdir, filename)))


class SearchOrderingTests(unittest.TestCase):
    def test_parallel_search_returns_query_and_rank_order(self) -> None:
        def fake_search(query: str, api_key: str, **_: object) -> list[dict]:
            return [
                {"url": f"{query}-2", "result_rank": 2},
                {"url": f"{query}-1", "result_rank": 1},
            ]

        with patch.object(search, "brave_llm_search", fake_search):
            results = search.search_parallel(["a", "b"], "key", max_workers=2)

        self.assertEqual([r["url"] for r in results], ["a-1", "a-2", "b-1", "b-2"])
        self.assertEqual([r["query_index"] for r in results], [0, 0, 1, 1])

    def test_parallel_search_returns_immediately_when_cancelled(self) -> None:
        with patch.object(search, "brave_llm_search") as mock_search:
            results = search.search_parallel(["a", "b"], "key", should_cancel=lambda: True)

        self.assertEqual(results, [])
        mock_search.assert_not_called()


class ImageSearchTests(unittest.TestCase):
    def test_brave_image_search_defaults_to_four_display_results(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "results": [
                        {
                            "title": f"Result {idx}",
                            "url": f"https://example.com/{idx}",
                            "thumbnail": {"src": f"https://cdn.example.com/{idx}.jpg"},
                            "properties": {
                                "url": f"https://cdn.example.com/{idx}.jpg",
                                "width": 1200 - idx,
                                "height": 800,
                            },
                            "meta_url": {"hostname": "example.com"},
                        }
                        for idx in range(1, 7)
                    ]
                }

        with patch.object(search.requests, "get", return_value=FakeResponse()) as mock_get:
            results = search.brave_image_search("any search", "key")

        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["count"], 12)
        self.assertEqual(len(results), 4)

    def test_brave_image_search_uses_safesearch_off_and_normalizes(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "results": [
                        {
                            "title": "Movie poster",
                            "url": "https://example.com/poster",
                            "thumbnail": {"src": "https://cdn.example.com/thumb.jpg"},
                            "properties": {
                                "url": "https://cdn.example.com/full.jpg",
                                "width": 640,
                                "height": 480,
                            },
                            "meta_url": {"hostname": "example.com"},
                        }
                    ]
                }

        with patch.object(search.requests, "get", return_value=FakeResponse()) as mock_get:
            results = search.brave_image_search("movie poster", "key", count=4)

        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["safesearch"], "off")
        self.assertEqual(kwargs["params"]["count"], 12)
        self.assertEqual(results[0]["thumbnail_url"], "https://cdn.example.com/thumb.jpg")
        self.assertEqual(results[0]["image_url"], "https://cdn.example.com/full.jpg")
        self.assertEqual(results[0]["source_domain"], "example.com")

    def test_brave_image_search_ranks_quality_before_returning_display_count(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "results": [
                        {
                            "title": "Tiny first result",
                            "url": "https://example.com/tiny",
                            "thumbnail": {"src": "https://cdn.example.com/tiny-thumb.jpg"},
                            "properties": {
                                "url": "https://cdn.example.com/tiny.jpg",
                                "width": 100,
                                "height": 100,
                            },
                            "meta_url": {"hostname": "example.com"},
                        },
                        {
                            "title": "Large second result",
                            "url": "https://example.com/large",
                            "thumbnail": {"src": "https://cdn.example.com/large-thumb.jpg"},
                            "properties": {
                                "url": "https://cdn.example.com/large.jpg",
                                "width": 1600,
                                "height": 1000,
                            },
                            "meta_url": {"hostname": "example.com"},
                        },
                    ]
                }

        with patch.object(search.requests, "get", return_value=FakeResponse()):
            results = search.brave_image_search("movie poster", "key", count=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Large second result")

    def test_brave_image_search_filters_cross_site_duplicates(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "results": [
                        {
                            "title": "Cyberpunk Edgerunners poster key art",
                            "url": "https://site-one.example/poster",
                            "thumbnail": {"src": "https://cdn.one.example/cyberpunk-edgerunners-key-art-300x200.jpg"},
                            "properties": {
                                "url": "https://cdn.one.example/cyberpunk-edgerunners-key-art.jpg",
                                "width": 1200,
                                "height": 800,
                            },
                            "meta_url": {"hostname": "site-one.example"},
                        },
                        {
                            "title": "Cyberpunk Edgerunners poster key art",
                            "url": "https://site-two.example/poster",
                            "thumbnail": {"src": "https://cdn.two.example/cyberpunk-edgerunners-key-art-300x200.jpg"},
                            "properties": {
                                "url": "https://cdn.two.example/cyberpunk-edgerunners-key-art.jpg",
                                "width": 900,
                                "height": 600,
                            },
                            "meta_url": {"hostname": "site-two.example"},
                        },
                        {
                            "title": "Cyberpunk Edgerunners character design Lucy",
                            "url": "https://site-three.example/lucy",
                            "thumbnail": {"src": "https://cdn.three.example/lucy-design.jpg"},
                            "properties": {
                                "url": "https://cdn.three.example/lucy-design.jpg",
                                "width": 1000,
                                "height": 700,
                            },
                            "meta_url": {"hostname": "site-three.example"},
                        },
                    ]
                }

        with patch.object(search.requests, "get", return_value=FakeResponse()):
            results = search.brave_image_search("anime poster", "key", count=4)

        self.assertEqual(
            [result["title"] for result in results],
            [
                "Cyberpunk Edgerunners poster key art",
                "Cyberpunk Edgerunners character design Lucy",
            ],
        )

    def test_brave_image_search_excludes_blocked_domains(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "results": [
                        {
                            "title": "TikTok video thumbnail",
                            "url": "https://www.tiktok.com/@user/video/123",
                            "thumbnail": {"src": "https://p16-sign.tiktokcdn.com/thumb1.jpg"},
                            "properties": {
                                "url": "https://p16-sign.tiktokcdn.com/full1.jpg",
                                "width": 720,
                                "height": 1280,
                            },
                            "meta_url": {"hostname": "www.tiktok.com"},
                        },
                        {
                            "title": "Another TikTok clip",
                            "url": "https://vm.tiktok.com/456",
                            "thumbnail": {"src": "https://p16-sign.tiktokcdn.com/thumb2.jpg"},
                            "properties": {
                                "url": "https://p16-sign.tiktokcdn.com/full2.jpg",
                                "width": 720,
                                "height": 1280,
                            },
                            "meta_url": {"hostname": "vm.tiktok.com"},
                        },
                    ] + [
                        {
                            "title": f"Good result {idx}",
                            "url": f"https://example.com/img/{idx}",
                            "thumbnail": {"src": f"https://cdn.example.com/thumb{idx}.jpg"},
                            "properties": {
                                "url": f"https://cdn.example.com/full{idx}.jpg",
                                "width": 1200,
                                "height": 800,
                            },
                            "meta_url": {"hostname": "example.com"},
                        }
                        for idx in range(1, 6)
                    ],
                }

        with patch.object(search.requests, "get", return_value=FakeResponse()):
            results = search.brave_image_search("test query", "key")

        self.assertEqual(len(results), 4)
        for result in results:
            domain = result.get("source_domain", "")
            self.assertFalse(
                domain.endswith("tiktok.com"),
                f"Blocked domain {domain!r} should not appear in results",
            )


class SourceRegistryTests(unittest.TestCase):
    def test_scoring_orders_context_and_includes_page_excerpt(self) -> None:
        registry = SourceRegistry()
        registry.add({
            "url": "https://low.example/a",
            "title": "Cooking notes",
            "domain": "low.example",
            "date": "2020",
            "snippets": ["unrelated text"],
            "query_origin": "cooking",
            "query_index": 0,
            "result_rank": 1,
        })
        registry.add({
            "url": "https://apple.com/news",
            "title": "Apple WWDC announcements",
            "domain": "apple.com",
            "date": "2026",
            "snippets": ["Apple announced iOS and macOS updates at WWDC"],
            "query_origin": "Apple WWDC 2026",
            "query_index": 0,
            "result_rank": 2,
        })

        registry.score_sources(
            "What did Apple announce at WWDC 2026?",
            ["Apple WWDC 2026"],
            "2026-06-11",
        )
        registry.set_page_excerpt(2, "Full article text about Apple WWDC.", "fetched")

        context = registry.format_knowledge_context()
        self.assertLess(context.index("[Source 2]"), context.index("[Source 1]"))
        self.assertIn("Page excerpt:", context)
        self.assertIn("Full article text", context)

class FetchPageExtractionTests(unittest.TestCase):
    def test_fetch_page_excerpt_reports_anti_bot_challenge(self) -> None:
        class FakeResponse:
            status_code = 202
            headers = {"content-type": "text/html; charset=UTF-8"}
            encoding = "utf-8"
            apparent_encoding = "utf-8"

            def __enter__(self):
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                del chunk_size
                yield b"<html><script>window.gokuProps = {}</script></html>"

        with patch.object(fetch_pages.requests, "get", return_value=FakeResponse()):
            excerpt, status = fetch_pages.fetch_page_excerpt(
                "https://www.imdb.com/title/tt0944947/episodes/?season=1",
                "IMDb episode ratings",
            )

        self.assertEqual(excerpt, "")
        self.assertEqual(status, "blocked by site anti-bot challenge")

    def test_trafilatura_is_used_when_available(self) -> None:
        fake = SimpleNamespace(
            extract=lambda *args, **kwargs: "Richer article text " * 20
        )
        original = sys.modules.get("trafilatura")
        sys.modules["trafilatura"] = fake
        try:
            text = fetch_pages._extract_text("<html><body>fallback</body></html>")
        finally:
            if original is None:
                sys.modules.pop("trafilatura", None)
            else:
                sys.modules["trafilatura"] = original

        self.assertIn("Richer article text", text)

    def test_stdlib_extractor_skips_navigation_and_scripts(self) -> None:
        html = """
        <html><body>
          <nav>Menu</nav>
          <article><p>Apple announced iOS and macOS updates at WWDC 2026.</p></article>
          <script>bad()</script>
        </body></html>
        """

        text = fetch_pages._extract_text(html)
        self.assertIn("Apple announced", text)
        self.assertNotIn("Menu", text)
        self.assertNotIn("bad", text)

    def test_fetch_page_excerpts_returns_immediately_when_cancelled(self) -> None:
        sources = [{"id": 1, "url": "https://example.com"}]

        with patch.object(fetch_pages, "fetch_page_excerpt") as mock_fetch:
            fetched = fetch_pages.fetch_page_excerpts(
                sources,
                "query",
                should_cancel=lambda: True,
            )

        self.assertEqual(fetched, {})
        mock_fetch.assert_not_called()

class BrowserProfileTests(unittest.TestCase):
    """Tests for the unified browser profile headers and retry logic."""

    _REQUIRED_KEYS = {
        "User-Agent", "Accept", "Accept-Language",
        "Upgrade-Insecure-Requests", "Sec-Fetch-Dest", "Sec-Fetch-Mode",
        "Sec-Fetch-Site", "Sec-Fetch-User", "Referer", "DNT",
    }
    _CHROME_HINT_KEYS = {"Sec-Ch-Ua", "Sec-Ch-Ua-Mobile", "Sec-Ch-Ua-Platform"}

    def test_all_profiles_contain_required_keys(self) -> None:
        for idx, profile in enumerate(fetch_pages._BROWSER_PROFILES):
            with self.subTest(profile_index=idx):
                self.assertTrue(
                    self._REQUIRED_KEYS.issubset(profile.keys()),
                    f"Profile {idx} missing: {self._REQUIRED_KEYS - profile.keys()}",
                )

    def test_chrome_profiles_include_client_hints(self) -> None:
        chrome = [p for p in fetch_pages._BROWSER_PROFILES if "Chrome" in p["User-Agent"]]
        self.assertGreater(len(chrome), 0, "No Chrome profiles found")
        for profile in chrome:
            for key in self._CHROME_HINT_KEYS:
                self.assertIn(key, profile, f"Chrome profile missing {key}")
            # Version in Sec-Ch-Ua must match the User-Agent Chrome version.
            ua_version = profile["User-Agent"].split("Chrome/")[1].split(".")[0]
            self.assertIn(
                f'"Google Chrome";v="{ua_version}"', profile["Sec-Ch-Ua"],
            )

    def test_firefox_profiles_omit_client_hints(self) -> None:
        firefox = [p for p in fetch_pages._BROWSER_PROFILES if "Firefox" in p["User-Agent"]]
        self.assertGreater(len(firefox), 0, "No Firefox profiles found")
        for profile in firefox:
            hint_keys = {k for k in profile if k.startswith("Sec-Ch-")}
            self.assertEqual(hint_keys, set(), f"Firefox profile has Client Hints: {hint_keys}")

    def test_no_profile_contains_accept_encoding(self) -> None:
        for idx, profile in enumerate(fetch_pages._BROWSER_PROFILES):
            self.assertNotIn(
                "Accept-Encoding", profile,
                f"Profile {idx} contains Accept-Encoding (brotli decode risk)",
            )

    def test_request_headers_returns_valid_profile(self) -> None:
        headers = fetch_pages._request_headers()
        self.assertIn(headers, fetch_pages._BROWSER_PROFILES)

    def test_request_headers_can_vary(self) -> None:
        seen = set()
        for _ in range(50):
            seen.add(id(fetch_pages._request_headers()))
        self.assertGreater(len(seen), 1, "All 50 calls returned the same profile")

    def test_retry_recovers_from_transient_403(self) -> None:
        good_html = "<html><body>" + ("<p>Paragraph about testing retry logic. " * 20) + "</p></body></html>"
        call_count = 0

        class BlockedResponse:
            status_code = 403
            headers = {"content-type": "text/html"}
            def __enter__(self): return self
            def __exit__(self, *_): return None

        class SuccessResponse:
            status_code = 200
            headers = {"content-type": "text/html; charset=UTF-8"}
            encoding = "utf-8"
            apparent_encoding = "utf-8"
            def __enter__(self): return self
            def __exit__(self, *_): return None
            def raise_for_status(self): return None
            def iter_content(self, chunk_size=8192): yield good_html.encode()

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return BlockedResponse() if call_count == 1 else SuccessResponse()

        with patch.object(fetch_pages.requests, "get", side_effect=side_effect), \
             patch.object(fetch_pages.time, "sleep"):
            excerpt, status = fetch_pages.fetch_page_excerpt(
                "https://example.com/article", "testing retry logic",
            )

        self.assertEqual(status, "fetched")
        self.assertGreater(len(excerpt), 0)
        self.assertEqual(call_count, 2)

    def test_persistent_403_returns_blocked_status(self) -> None:
        call_count = 0

        class BlockedResponse:
            status_code = 403
            headers = {"content-type": "text/html"}
            def __enter__(self): return self
            def __exit__(self, *_): return None

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return BlockedResponse()

        with patch.object(fetch_pages.requests, "get", side_effect=side_effect), \
             patch.object(fetch_pages.time, "sleep"):
            excerpt, status = fetch_pages.fetch_page_excerpt(
                "https://example.com/article", "test query",
            )

        self.assertEqual(excerpt, "")
        self.assertEqual(status, "blocked by site: HTTP 403")
        self.assertEqual(call_count, 2, "Should attempt exactly 2 requests (initial + 1 retry)")


class PipelineHelperTests(unittest.TestCase):
    def test_query_list_coercion_rejects_non_lists_and_caps_results(self) -> None:
        self.assertEqual(researcher._coerce_query_list("abc", 2), [])
        self.assertEqual(
            researcher._coerce_query_list([" one ", "", "one", "two", "three"], 2),
            ["one", "two"],
        )


class PipelineRestructureTests(unittest.TestCase):
    class FakeLLM:
        def __init__(self, *_: object, **__: object) -> None:
            self.thinking_budget = 0
            self.ask_json_calls = 0

        def ask_json(self, *_: object, **__: object) -> dict:
            self.ask_json_calls += 1
            raise AssertionError("Quick mode should not call LLM planning")

        def ask_text_stream(self, *_: object, **__: object):
            yield {"type": "content", "delta": "Answer [1]"}
            yield {"type": "done", "content": "Answer [1]", "thinking": ""}

    def test_quick_mode_uses_direct_query_without_planning_llm(self) -> None:
        events: list[dict] = []

        def fake_search_parallel(queries: list[str], *_: object, **__: object):
            return [{
                "url": "https://example.com",
                "title": "Question answer",
                "domain": "example.com",
                "snippets": ["Question answer details"],
                "date": "2026",
                "query_origin": queries[0],
                "query_index": 0,
                "result_rank": 1,
            }]

        provider_cfg = {
            "name": "Fake",
            "base_url": "https://example.com",
            "model": "fake",
            "api_key": "key",
        }

        with patch.object(researcher, "get_provider_config", return_value=provider_cfg), \
             patch.object(researcher, "get_brave_api_key", return_value="key"), \
             patch.object(researcher, "LLMClient", self.FakeLLM), \
             patch.object(researcher, "search_parallel", fake_search_parallel), \
             patch.object(researcher, "fetch_page_excerpts", return_value={}):
            result = researcher.research_stream(
                "Question?",
                preset_name="quick",
                include_images=False,
                on_event=events.append,
            )

        query_events = [event for event in events if event.get("type") == "queries"]
        self.assertEqual(result, "Answer [1]")
        self.assertEqual(query_events[0]["queries"], ["Question?"])
        self.assertEqual(query_events[0]["label"], "Direct search query")

    def test_deterministic_coverage_generates_followups_for_weak_queries(self) -> None:
        registry = SourceRegistry()
        registry.add({
            "url": "https://example.com/alpha",
            "title": "Alpha release notes",
            "domain": "example.com",
            "snippets": ["Alpha includes a documented release schedule."],
            "query_origin": "alpha release schedule",
            "result_rank": 1,
        })

        result = researcher._deterministic_coverage_analysis(
            "Compare alpha release timing with gamma patent litigation",
            ["alpha release schedule", "gamma patent litigation"],
            registry,
            max_followups=2,
            previous_queries=["alpha release schedule", "gamma patent litigation"],
        )

        statuses = {row["question"]: row["status"] for row in result["answered"]}
        self.assertEqual(statuses["alpha release schedule"], "answered")
        self.assertIn(statuses["gamma patent litigation"], {"partial", "unanswered"})
        self.assertTrue(result["followup_queries"])

    def test_deterministic_coverage_scores_blocks_individually(self) -> None:
        registry = SourceRegistry()
        # Source with terms in different paragraphs
        sid = registry.add({
            "url": "https://example.com/split",
            "title": "Split paragraphs",
            "domain": "example.com",
            "query_origin": "alpha beta gamma delta",
            "result_rank": 1,
        })
        registry.set_page_excerpt(sid, "Paragraph one has alpha and beta.\n\nParagraph two has gamma and delta.", "fetched")

        result = researcher._deterministic_coverage_analysis(
            "alpha beta gamma delta",
            ["alpha beta gamma delta"],
            registry,
            max_followups=2,
            previous_queries=["alpha beta gamma delta"],
        )
        
        row = result["answered"][0]
        # Query has 4 terms. Max block has 2. Base score 2/4 = 0.5
        # Since it's query_origin, +0.01 -> 0.51
        self.assertAlmostEqual(row["score"], 0.51, places=2)
        self.assertEqual(row["status"], "partial")


class CancellationTests(unittest.TestCase):
    def test_cancel_helper_raises_when_requested(self) -> None:
        with self.assertRaises(SearchCancelled):
            _raise_if_cancelled(lambda: True)

    def test_cancel_endpoint_sets_event(self) -> None:
        cancel_event = threading.Event()
        queue: Queue = Queue()
        app_module._active_searches["test-search"] = {
            "queue": queue,
            "cancel_event": cancel_event,
        }
        try:
            client = app_module.app.test_client()
            response = client.post("/api/search/test-search/cancel")
        finally:
            app_module._active_searches.pop("test-search", None)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(cancel_event.is_set())
        self.assertEqual(queue.get_nowait()["type"], "status")

    def test_completed_active_searches_are_cleaned_after_ttl(self) -> None:
        app_module._active_searches["stale-search"] = {
            "queue": Queue(),
            "cancel_event": threading.Event(),
            "done": True,
            "completed_at": time.monotonic()
            - app_module._COMPLETED_SEARCH_TTL_SECONDS
            - 1,
        }
        try:
            app_module._cleanup_active_searches()
            self.assertNotIn("stale-search", app_module._active_searches)
        finally:
            app_module._active_searches.pop("stale-search", None)

    def test_search_endpoint_accepts_provider_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            original_history = app_module._history
            app_module._history = SearchHistory(output_dir=tmpdir)
            done_event = threading.Event()

            def fake_research_stream(*, on_event, **_: object) -> str:
                on_event({
                    "type": "done",
                    "content": "Answer",
                    "thinking": "",
                    "sources": [],
                    "images": [],
                })
                done_event.set()
                return "Answer"

            try:
                client = app_module.app.test_client()
                with patch.object(app_module, "research_stream", fake_research_stream):
                    response = client.post(
                        "/api/search",
                        json={
                            "question": "Question?",
                            "provider": "deepseek",
                            "mode": "quick",
                        },
                    )
                self.assertTrue(done_event.wait(timeout=1))
            finally:
                app_module._history = original_history
                for search_id in list(app_module._active_searches):
                    app_module._active_searches.pop(search_id, None)

        self.assertEqual(response.status_code, 200)


class TextHelpersTests(unittest.TestCase):
    def test_stem_word_strips_common_suffixes(self) -> None:
        from text_utils import stem_word
        self.assertEqual(stem_word("Running"), "runn")
        self.assertEqual(stem_word("walked"), "walk")
        self.assertEqual(stem_word("buses"), "bus")
        self.assertEqual(stem_word("apples"), "apple")
        self.assertEqual(stem_word("states"), "state")
        self.assertEqual(stem_word("boxes"), "box")
        self.assertEqual(stem_word("cats"), "cat")
        self.assertEqual(stem_word("boss"), "boss")
        self.assertEqual(stem_word("quickly"), "quick")
        self.assertEqual(stem_word("management"), "manage")
        self.assertEqual(stem_word("sing"), "sing")
        self.assertEqual(stem_word("king"), "king")
        self.assertEqual(stem_word("red"), "red")
        self.assertEqual(stem_word("fly"), "fly")

    def test_tokenize_terms_filters_expanded_stop_words(self) -> None:
        from text_utils import tokenize_terms
        text = "This is a quick test to find the latest information about Apple."
        terms = tokenize_terms(text)
        self.assertNotIn("this", terms)
        self.assertNotIn("is", terms)
        self.assertNotIn("find", terms)
        self.assertNotIn("latest", terms)
        self.assertNotIn("information", terms)
        self.assertIn("quick", terms)
        self.assertIn("test", terms)
        self.assertIn("apple", terms)


class MistralIntegrationTests(unittest.TestCase):
    def test_mistral_config_and_aliases(self) -> None:
        from config import MODEL_PROVIDERS, _PROVIDER_ALIASES
        self.assertIn("mistral-medium-3.5", MODEL_PROVIDERS)
        mistral_config = MODEL_PROVIDERS["mistral-medium-3.5"]
        self.assertEqual(mistral_config["name"], "Mistral Medium 3.5")
        self.assertEqual(mistral_config["model"], "mistral-medium-3.5")
        self.assertEqual(mistral_config["thinking_style"], "mistral")
        self.assertTrue(mistral_config["supports_thinking"])
        self.assertTrue(mistral_config["supports_json_mode"])
        
        self.assertIn("mistral", _PROVIDER_ALIASES)
        self.assertEqual(_PROVIDER_ALIASES["mistral"], "mistral-medium-3.5")

    def test_mistral_thinking_params(self) -> None:
        from llm import _THINKING_BUILDERS
        self.assertIn("mistral", _THINKING_BUILDERS)
        builder = _THINKING_BUILDERS["mistral"]
        self.assertEqual(builder(0), {"reasoning_effort": "none"})
        self.assertEqual(builder(-5), {"reasoning_effort": "none"})
        self.assertEqual(builder(100), {"reasoning_effort": "high"})
        self.assertEqual(builder(2048), {"reasoning_effort": "high"})

    def test_extract_response_with_nested_thinking(self) -> None:
        from llm import _extract_response
        
        # Test case 1: Standard string thinking (DeepSeek/Gemini style or standard)
        data_standard = {
            "choices": [{
                "message": {
                    "content": "Hello",
                    "reasoning_content": "Initial thoughts"
                }
            }]
        }
        content, thinking = _extract_response(data_standard)
        self.assertEqual(content, "Hello")
        self.assertEqual(thinking, "Initial thoughts")

        # Test case 2: List-nested block thinking with list of dicts (Mistral style)
        data_mistral_list = {
            "choices": [{
                "message": {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [
                                {"type": "text", "text": "Thought part 1. "},
                                {"type": "text", "text": "Thought part 2."}
                            ]
                        },
                        {
                            "type": "text",
                            "text": "Actual answer content."
                        }
                    ]
                }
            }]
        }
        content, thinking = _extract_response(data_mistral_list)
        self.assertEqual(content, "Actual answer content.")
        self.assertEqual(thinking, "Thought part 1. Thought part 2.")

        # Test case 3: List-nested block thinking with dict
        data_dict = {
            "choices": [{
                "message": {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": {"text": "Simple thought"}
                        },
                        {
                            "type": "text",
                            "text": "Answer"
                        }
                    ]
                }
            }]
        }
        content, thinking = _extract_response(data_dict)
        self.assertEqual(content, "Answer")
        self.assertEqual(thinking, "Simple thought")

        # Test case 4: List-nested block thinking with string
        data_str = {
            "choices": [{
                "message": {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Simple string thought"
                        },
                        {
                            "type": "text",
                            "text": "Answer"
                        }
                    ]
                }
            }]
        }
        content, thinking = _extract_response(data_str)
        self.assertEqual(content, "Answer")
        self.assertEqual(thinking, "Simple string thought")

    @patch("llm.requests.post")
    def test_chat_completion_stream_with_nested_thinking(self, mock_post) -> None:
        from llm import LLMClient
        
        # Setup mock stream response
        class FakeStreamResponse:
            def __init__(self, lines: list[str]) -> None:
                self.lines = lines
                self.status_code = 200
            def raise_for_status(self) -> None:
                pass
            def close(self) -> None:
                pass
            def iter_lines(self, *args, **kwargs):
                return iter(self.lines)

        sse_lines = [
            "data: {\"choices\": [{\"delta\": {\"content\": [{\"type\": \"thinking\", \"thinking\": [{\"type\": \"text\", \"text\": \"Mistral thinking part 1. \"}]}]}}]}",
            "data: {\"choices\": [{\"delta\": {\"content\": [{\"type\": \"thinking\", \"thinking\": [{\"type\": \"text\", \"text\": \"Mistral thinking part 2.\"}]}]}}]}",
            "data: {\"choices\": [{\"delta\": {\"content\": [{\"type\": \"text\", \"text\": \"Main response content.\"}]}}]}",
            "data: [DONE]"
        ]
        
        mock_post.return_value = FakeStreamResponse(sse_lines)
        
        provider_cfg = {
            "name": "Mistral Medium 3.5",
            "base_url": "https://api.mistral.ai/v1",
            "model": "mistral-medium-3.5",
            "api_key": "fake-key",
            "supports_json_mode": True,
            "supports_thinking": True,
            "thinking_style": "mistral"
        }
        
        client = LLMClient(provider_cfg, thinking_budget=2048)
        events = list(client.ask_text_stream("test query"))
        
        # Verify events emitted
        # We expect two thinking events, one content event, and one done event.
        self.assertEqual(len(events), 4)
        
        self.assertEqual(events[0], {"type": "thinking", "delta": "Mistral thinking part 1. "})
        self.assertEqual(events[1], {"type": "thinking", "delta": "Mistral thinking part 2."})
        self.assertEqual(events[2], {"type": "content", "delta": "Main response content."})
        self.assertEqual(events[3], {
            "type": "done",
            "content": "Main response content.",
            "thinking": "Mistral thinking part 1. Mistral thinking part 2."
        })


if __name__ == "__main__":
    unittest.main()

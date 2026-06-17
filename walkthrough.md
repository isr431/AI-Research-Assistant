# AI Research Assistant — Architecture Walkthrough

## Overview

The AI Research Assistant is a real-time web application that performs source-backed research using LLMs. It features SSE-based streaming, cancellable searches, search history management, markdown rendering with interactive citations, model thinking/reasoning displays, multi-model support, deterministic coverage checks, full-page source enrichment, lightweight source scoring, and optional side-channel image results.

The product UI is implemented as a Swiss technical data interface: light mode, high-density layout, square geometry, thin grid-like borders, pure black typography, restrained gray surfaces, and a single matte red accent. It favors precision and legibility over decorative depth.

---

## Backend Components

### [app.py](app.py) — Flask Web Server
The entry point for the web application. Features:
- Serves the Single-Page Application (SPA) on `/`.
- Provides an SSE (Server-Sent Events) endpoint at `/api/search/<id>/stream` which yields `data:` chunks from a thread-safe `queue.Queue`.
- Creates background threads for research to prevent blocking the web server.
- Provides `/api/search/<id>/cancel` to request cooperative cancellation of a running search.
- Contains REST endpoints for fetching providers and managing search history.
- Accepts an `include_images` search option and forwards image result payloads through SSE/history without coupling them to answer synthesis.

### [config.py](config.py) — Configuration & Multi-model Support
Handles API keys, global configurations, and model definitions.
- `MODEL_PROVIDERS`: A flat dictionary containing specific models (e.g., `deepseek-v4-pro`, `deepseek-v4-flash`, `gemini-3.5-flash`, `mistral-medium-3.5`), their URLs, and capabilities.
- Each model configuration defines `thinking_presets` that dictate the exact reasoning/thinking level passed onto the API based on the active search mode (Quick, Moderate, or Deep).
- Configuration errors raise normal Python exceptions so the web stream can surface failures cleanly instead of terminating a worker thread.
- `SEARCH_PRESETS`: Defines three performance modes:
  - **Quick**: One direct search pass and one concise answer pass. It skips planning and refinement to preserve latency.
  - **Moderate**: One planned search pass, deterministic coverage checking, at most one follow-up pass, and one detailed synthesis pass.
  - **Deep**: Broader planned search, deterministic multi-pass coverage checking, and one final report-style synthesis.
- Presets also control top-source full-page enrichment: Quick fetches fewer page excerpts than Moderate/Deep to preserve latency.

### [history.py](history.py) — Search History Manager
Maintains a thread-safe JSON index (`output/history.json`) alongside timestamped markdown files. Features:
- Uses a `threading.RLock` to prevent race conditions during concurrent file I/O operations (saving/deleting).
- Saves response content and model thinking.
- Embeds structured source data as hidden HTML comments in markdown (`<!-- SOURCES_JSON ... -->`).
- Embeds optional image result data as hidden HTML comments (`<!-- IMAGES_JSON ... -->`) so saved searches can rehydrate the image panel.
- Embeds pipeline metadata as hidden HTML comments (`<!-- PIPELINE_METADATA_JSON ... -->`), including query events, deterministic coverage results, source-read events, and page-fetch status for collected sources.
- Adds YAML front matter with metadata for each search.

### [llm.py](llm.py) — LLM Client & Stream Processing
OpenAI-compatible client with robust streaming and JSON extraction capabilities.
- `chat_completion_stream()`: A highly specialized SSE parser that connects to LLMs via `stream=True`. It yields structured event dicts: `{"type": "thinking"|"content"|"done", ...}`.
- Implements a state machine to parse Gemini's inline `<thought>...</thought>` tags natively, stripping them from the main content stream and yielding them as discrete `thinking` events.
- Handles Mistral-specific and DeepSeek-specific reasoning/thinking mode using dynamic string-based parameter builders that map abstract levels to specific API parameters (`reasoning_effort` and `thinking_level`).
- Handles automated retries for malformed JSON when `json_mode=True`.

### [prompts.py](prompts.py) — System Instructions
Centralized location for all prompt templates used across the pipeline:
- `PLAN_PROMPT`: Instructs the LLM to produce focused web search queries.
- Synthesis Prompts: Mode-specific prompts (`SYNTHESIS_PROMPT_QUICK`, `SYNTHESIS_PROMPT_MODERATE`, `SYNTHESIS_PROMPT_DEEP`) that instruct the LLM on formatting (e.g., concise paragraphs vs formal reports) and strictly forbid em dashes.

### [researcher.py](researcher.py) — Event-Driven Pipeline
The core orchestrator. Exposes `research_stream()` which coordinates:
1. **Planning**: Quick mode uses the original question directly. Moderate and Deep call the LLM to generate focused search queries.
2. **Search**: Passes sub-queries to the Brave Search API and registers results in a source registry.
3. **Source Scoring**: Scores sources using title/snippet/query overlap, freshness, primary-source signals, and search rank before formatting context.
4. **Full-Page Enrichment**: Fetches readable excerpts for the top scored sources and adds them to the model context when extraction succeeds.
5. **Coverage Checks**: Quick mode emits a single-pass coverage summary. Moderate/Deep modes run deterministic token-overlap coverage checks using block-level density matching against individual titles, snippets, and paragraph excerpts, then trigger capped follow-up searches for weakly covered areas.
6. **Synthesis**: Streams the answer back to the user using the mode-specific synthesis prompt.
7. **Validation**: Removes phantom citations and saves only cited sources in the final result payload.

Image results are optional and intentionally separate from the LLM context. When enabled, the pipeline starts a background image search from the original user question and returns up to four image results. Image search fails soft so text research is not blocked by image API errors.

### [fetch_pages.py](fetch_pages.py) — Full-Page Excerpt Fetching
Fetches compact readable excerpts for the highest-scored sources.
- Rotates through unified browser profiles (Chrome/Firefox × Windows/macOS/Linux), each coupling a User-Agent with its correct companion headers (e.g. Chrome Client Hints are present only on Chrome profiles). A random profile is selected per request to avoid sharing a single fingerprint across the fetch batch.
- Sends high-fidelity browser headers across all profiles: modern `Accept`, `Accept-Language`, `Sec-Fetch-*`, Google `Referer`, and `DNT`. Omits `Accept-Encoding` to let `requests` negotiate compression safely based on installed decoders.
- Retries once with a different browser profile on HTTP 403/429, after a short random backoff (1–3 s). Content-based blocks (e.g., AWS WAF JS challenges, "verify you are human") are not retried. Avoids overly generic markers to ensure compatibility with MediaWiki inline JSON payloads.
- Adds a small random jitter (0–0.5 s) before each request to avoid burst-traffic detection by CDNs.
- Uses `trafilatura` when installed for higher-quality article extraction, with a standard-library `HTMLParser` fallback that skips scripts, styles, navigation, forms, SVG, and footer content.
- Selects query-relevant excerpts rather than passing entire pages into the context.
- Fails soft: unsupported content types, network errors, and extraction failures leave the original Brave snippets intact.
- Supports cooperative cancellation while fetching multiple top-source pages.

### [search.py](search.py) — Brave Search Client
Handles concurrent web searches using Brave Search LLM Context API and optional Brave Image Search.
- Leverages `ThreadPoolExecutor` to execute multiple sub-queries in parallel.
- Automatically retries transient errors (429, 5xx) with a backoff strategy.
- Adds per-query result rank and restores deterministic ordering after parallel execution, so source IDs are stable by original query order and within-query rank.
- Normalizes search result fields before passing them to the `SourceRegistry`.
- Calls Brave Image Search with `safesearch=off`, overfetches a small candidate pool, ranks image candidates by metadata completeness, dimensions, aspect ratio, and Brave rank, then returns up to four display results.
- Filters image duplicates conservatively using normalized image/thumbnail URLs and stable filename signatures across different hosts/CDNs.
- Blocks undesirable image domains (e.g., `tiktok.com`) during the candidate collection phase, ensuring they do not occupy output slots while overfetching guarantees that the target count of 4 images is still met.

### [sources.py](sources.py) — Citation & Source Management
Manages the deduplication and validation of sources:
- `SourceRegistry`: Assigns incrementing `[N]` IDs to unique URLs, strips tracking parameters, merges duplicate snippets, preserves query origins, and formats rich source blocks with title, URL, domain, date, query origin, and snippets.
- `score_sources()`: Applies lightweight relevance scoring so higher-value sources appear earlier in the LLM context and low-scored sources are trimmed first when context is over budget.
- `top_sources_for_fetch()` / `set_page_excerpt()`: Select top sources for full-page enrichment and attach fetched excerpts or fetch status to source records.
- `validate_citations()`: Strips phantom citations that point to nonexistent source IDs.
- `cited_source_ids()`: Identifies cited sources so final history/UI payloads include only sources actually referenced in the answer.

---

## Frontend Components

### [static/index.html](static/index.html) — HTML Structure
The layout consists of a fixed 280px sidebar and a main research workspace.
- Uses a compact left rail for the square knowledge-graph mark, model selector, new-search command, and search history.
- Adds a default-on Images toggle in the sidebar controls.
- Places the search input, square segmented mode control (Quick / Moderate / Deep), and arrow submit button in a single top bar.
- Renders research output as discrete technical panels: pipeline, model thinking, response, optional images, and sources.
- Uses mono-label section markers such as `// STANDBY`, `// PIPELINE`, `// IMAGES`, and `// SOURCES` to reinforce the data-console tone.

### [static/app.js](static/app.js) — Client-Side Application Logic
Vanilla JavaScript managing the Single-Page Application:
- Consumes the `EventSource` SSE stream.
- Reuses the search button as a cancel button while a search is running.
- Sends the Images toggle state with new searches and disables the toggle during an active stream.
- Post-processes `marked.js` output to identify `[N]` references in markdown text and convert them into interactive floating citation tooltips.
- Renders `images` SSE events into a compact separate image grid and rehydrates saved image results from history.
- Renders compact query chips, coverage summaries/follow-up queries, and source-read events inside the pipeline panel.
- Rehydrates saved pipeline metadata when viewing a history item.
- Throttles markdown rendering (150ms) to ensure smooth animations without locking the main thread.
- Dynamically calculates the position and width of the animated mode toggle slider based on the active button.
- Updates search history with relative time formatting and handles hover-delete logic.

### [static/style.css](static/style.css) — Swiss Technical Data Interface
A comprehensive 1,600+ line design system:
- Cool light-mode palette with off-white app chrome (`#F4F5F7`), white panels, gray inset wells, and pure black primary text.
- Single matte red accent (`#C8322B`) used for active states, focus rings, citations, the mode slider, and the square logo block.
- Flat containers with 1px borders, square corners, no glassmorphism, no radial gradients, and minimal shadow usage limited to small overlay affordances like citation tooltips and toasts.
- Typography built around Space Grotesk for interface text and JetBrains Mono for metadata, labels, chips, table headers, URLs, token counts, and technical status copy.
- Uniform monochrome mode badges and a square segmented mode selector, avoiding the older color-coded Quick / Moderate / Deep treatment.
- High-density panel styling for the pipeline, thinking stream, markdown response, image results, citations, source list, code blocks, tables, and blockquotes.
- Image tiles use fixed aspect ratios, metadata rows, and broken-image placeholders so thumbnails do not shift the layout.
- Restrained motion: short fades, a square pulse ring for the active pipeline stage, a red streaming cursor, spinner, and toast in/out transitions.
- Desktop-optimized design: Mobile responsive styles and breakpoints have been removed to keep the interface strictly focused on the core desktop knowledge console experience.

---

## How to Run

```bash
# Start the Web GUI
python3 app.py
# → Open http://127.0.0.1:31415
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The test suite covers metadata and image-history persistence, concurrent history writes, deterministic search ordering, fixed-count image search, image normalization/ranking/deduplication, blocked image domain filtering, source scoring and enrichment, page extraction fallback behavior, browser profile completeness and consistency (Chrome Client Hints present, Firefox Client Hints absent, no Accept-Encoding), profile selection randomness, retry recovery from transient 403/429 blocks, persistent block handling, quick-mode planning behavior, deterministic coverage helpers, provider configurations and aliases, Mistral thinking parameters, list-nested thinking content extraction under streaming/non-streaming parsers, active-search cleanup, and cancellation helpers.

# AI Research Assistant

A real-time, source-backed research tool powered by LLMs. Ask a question, get a cited answer — streamed live to your browser.

<p align="center">
  <img src="docs/screenshot.png" alt="AI Research Assistant UI" width="720" />
</p>

## What it does

AI Research Assistant takes a natural-language question, searches the web via the Brave Search API, scores and enriches the most relevant sources, then synthesises a fully-cited answer using an LLM of your choice. The entire pipeline — from query planning to final response — streams to the browser in real time through Server-Sent Events.

### Key capabilities

- **Three research modes** — *Quick* for fast answers, *Moderate* for balanced depth, *Deep* for comprehensive reports with multi-pass coverage checking.
- **Multi-model support** — switch between DeepSeek V4 (Pro / Flash) and Gemini 3.5 Flash from the UI.
- **Live streaming** — answers render token-by-token with a streaming cursor, model thinking/reasoning displayed in a collapsible panel.
- **Interactive citations** — `[N]` references become hoverable tooltips showing source title, domain, and snippet.
- **Full-page enrichment** — top-scored sources are fetched and distilled into query-relevant excerpts so the LLM gets richer context than search snippets alone.
- **Optional image results** — toggleable image search returns ranked, deduplicated thumbnails alongside the text answer.
- **Search history** — every research session is saved as markdown with embedded source and pipeline metadata, rehydratable from the sidebar.
- **Cancellable searches** — the search button transforms into a cancel button mid-stream for cooperative cancellation.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3 · Flask · SSE streaming |
| Frontend | Vanilla JS · marked.js · highlight.js |
| Search | Brave Web Search API · Brave Image Search API |
| LLM | OpenAI-compatible API (DeepSeek, Gemini) |
| Styling | Custom CSS — Swiss technical data interface |
| Fonts | Space Grotesk · JetBrains Mono |

---

## Getting started

### Prerequisites

- Python 3.10+
- A [Brave Search API](https://brave.com/search/api/) key
- At least one LLM API key (DeepSeek or Google Gemini)

### Installation

```bash
# Clone the repository
git clone https://github.com/isr431/ai-research-assistant.git
cd ai-research-assistant

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

```env
# Brave Search API
BRAVE_API_KEY=your-brave-api-key-here

# DeepSeek (default provider)
DEEPSEEK_API_KEY=your-deepseek-key-here

# Google Gemini (optional alternative)
# GEMINI_API_KEY=your-gemini-key-here
```

You only need one LLM provider key to get started. Uncomment and set `GEMINI_API_KEY` if you want Gemini as an option in the model selector.

### Run

```bash
python3 app.py
```

Open **http://127.0.0.1:31415** in your browser.

---

## Project structure

```
├── app.py            # Flask server, SSE endpoints, search lifecycle
├── config.py         # API keys, model providers, search presets
├── researcher.py     # Core pipeline orchestrator
├── search.py         # Brave Search client (web + image)
├── sources.py        # Source registry, scoring, citation validation
├── fetch_pages.py    # Full-page excerpt fetching & extraction
├── llm.py            # LLM streaming client & JSON extraction
├── prompts.py        # System prompt templates
├── history.py        # Thread-safe search history (JSON + markdown)
├── text_utils.py     # Text helper utilities
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── static/
│   ├── index.html    # Single-page application shell
│   ├── app.js        # Client-side logic, SSE consumer, rendering
│   └── style.css     # Swiss technical data interface (1600+ lines)
└── tests/
    └── test_pipeline.py
```

---

## How it works

```
User question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Query Plan  │────▶│ Brave Search │────▶│ Source Scoring  │
│  (LLM)       │     │ (parallel)   │     │ & Deduplication │
└─────────────┘     └──────────────┘     └────────┬───────┘
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │ Page Enrichment  │
                                         │ (top sources)    │
                                         └────────┬────────┘
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │ Coverage Check   │──── follow-up
                                         │ (Moderate/Deep)  │     searches
                                         └────────┬────────┘     if needed
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │ LLM Synthesis    │
                                         │ (streamed SSE)   │
                                         └────────┬────────┘
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │ Citation Valid.  │
                                         │ + History Save   │
                                         └─────────────────┘
```

1. **Planning** — Quick mode uses the question directly; Moderate and Deep ask the LLM to decompose it into focused sub-queries.
2. **Search** — Sub-queries are dispatched to the Brave Search API in parallel. Results are deduplicated and registered with stable IDs.
3. **Scoring** — Sources are ranked by title/snippet relevance, freshness, primary-source signals, and search rank.
4. **Enrichment** — The highest-scored sources are fetched and distilled into query-relevant excerpts (uses trafilatura when available, with an HTML parser fallback).
5. **Coverage** — Moderate and Deep modes run deterministic token-overlap checks and trigger capped follow-up searches for weak areas.
6. **Synthesis** — The LLM generates a cited answer streamed to the browser. Mode-specific prompts control depth and format.
7. **Validation** — Phantom citations are stripped; only sources actually referenced in the answer are included in the final output.

---

## Research modes

| Mode | Planning | Search passes | Coverage checks | Answer style |
|---|---|---|---|---|
| **Quick** | Direct question | 1 | Summary only | Concise paragraphs |
| **Moderate** | LLM-planned queries | 1 + up to 1 follow-up | Deterministic | Detailed answer |
| **Deep** | LLM-planned queries | 1 + multiple follow-ups | Multi-pass deterministic | Formal report |

---

## Tests

```bash
python3 -m unittest discover -s tests
```

The test suite covers history persistence, concurrent writes, deterministic search ordering, image search ranking/deduplication, source scoring, page extraction, coverage helpers, provider aliases, and cancellation.

---

## License

This project is provided as-is for personal and educational use.

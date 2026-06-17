"""Configuration: API keys, model providers, and search/research presets."""

import os
from dotenv import load_dotenv

# Load .env file if present (does not override existing env vars)
load_dotenv()

# ---------------------------------------------------------------------------
# Model Provider Configurations
# ---------------------------------------------------------------------------
# Each entry is a model option shown by the web UI and accepted by the API.

MODEL_PROVIDERS = {
    "deepseek-v4-pro": {
        "name": "DeepSeek V4 Pro",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "supports_json_mode": True,
        "supports_thinking": True,
        # DeepSeek uses {"thinking": {"type": "enabled"}} + "reasoning_effort"
        # Only "high" and "max" are effective; low/medium silently map to high.
        "thinking_style": "deepseek",
    },
    "deepseek-v4-flash": {
        "name": "DeepSeek V4 Flash",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "supports_json_mode": True,
        "supports_thinking": True,
        # DeepSeek uses {"thinking": {"type": "enabled"}} + "reasoning_effort"
        # Only "high" and "max" are effective; low/medium silently map to high.
        "thinking_style": "deepseek",
    },
    "gemini-3.5-flash": {
        "name": "Gemini 3.5 Flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "supports_json_mode": True,
        "supports_thinking": True,
        # Gemini uses the extra_body field with google.thinking_config.
        # include_thoughts=true causes thinking to be inlined in content
        # wrapped in <thought>...</thought> tags.
        "thinking_style": "gemini",
    },
    "mistral-medium-3.5": {
        "name": "Mistral Medium 3.5",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-medium-3.5",
        "api_key_env": "MISTRAL_API_KEY",
        "supports_json_mode": True,
        "supports_thinking": True,
        "thinking_style": "mistral",
    },
}

DEFAULT_PROVIDER = "deepseek-v4-flash"

# Backwards-compatible aliases for older API payloads or saved history.
_PROVIDER_ALIASES = {
    "deepseek": "deepseek-v4-flash",
    "gemini": "gemini-3.5-flash",
    "mistral": "mistral-medium-3.5",
}

# ---------------------------------------------------------------------------
# Search / Research Presets
# ---------------------------------------------------------------------------

SEARCH_PRESETS = {
    "quick": {
        "sub_queries": 1,
        "max_passes": 1,
        "urls_per_query": 8,
        "tokens_per_query": 4096,
        "output_style": "concise",
        "max_context_tokens": 14_000,
        "thinking_budget": 2048,
        "full_page_sources": 2,
        "total_full_page_sources": 2,
        "full_page_chars": 3500,
        "followup_queries_per_pass": 0,
    },
    "moderate": {
        "sub_queries": 3,
        "max_passes": 2,
        "urls_per_query": 6,
        "tokens_per_query": 6000,
        "output_style": "detailed",
        "max_context_tokens": 26_000,
        "thinking_budget": 8192,
        "synthesis_budget": 8192,
        "full_page_sources": 4,
        "total_full_page_sources": 5,
        "full_page_chars": 6000,
        "followup_queries_per_pass": 2,
    },
    "deep": {
        "sub_queries": 5,
        "max_passes": 3,
        "urls_per_query": 8,
        "tokens_per_query": 8192,
        "output_style": "report",
        "max_context_tokens": 40_000,
        "thinking_budget": 32768,
        "synthesis_budget": 32768,
        "full_page_sources": 6,
        "total_full_page_sources": 8,
        "full_page_chars": 8000,
        "followup_queries_per_pass": 2,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_provider_name(provider_name: str | None = None) -> str:
    """Return the canonical provider key for a user/API supplied provider name."""
    name = provider_name or DEFAULT_PROVIDER
    return _PROVIDER_ALIASES.get(name, name)


def get_provider_config(provider_name: str | None = None) -> dict:
    """Return the provider config dict, validating the API key is set.

    Accepts both full model keys (e.g. ``"deepseek-v4-flash"``) and
    short aliases (e.g. ``"deepseek"``).
    """
    name = normalize_provider_name(provider_name)
    if name not in MODEL_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{name}'. Choose from: {', '.join(MODEL_PROVIDERS)}"
        )

    cfg = MODEL_PROVIDERS[name]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(
            f"Missing API key for {cfg['name']}. "
            f"Set the {cfg['api_key_env']} environment variable "
            f"(or add it to your .env file)."
        )

    return {**cfg, "api_key": api_key}


def get_brave_api_key() -> str:
    """Return the Brave Search API key, raising a clear error if missing."""
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Missing BRAVE_API_KEY. "
            "Set the BRAVE_API_KEY environment variable "
            "(or add it to your .env file)."
        )
    return key

"""Thin wrapper around the OpenAI-compatible chat completions API.

Supports provider-specific thinking/reasoning modes for DeepSeek and Gemini.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

import requests

from prompts import RETRY_JSON_PROMPT


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Extract JSON from an LLM response, handling common quirks.

    Tries in order:
    1. Direct ``json.loads``
    2. Strip markdown fences (```json … ```)
    3. Extract the first ``{ … }`` block
    """
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First { … } block
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# Thinking-level mapping helpers
# ---------------------------------------------------------------------------

def _deepseek_thinking_params(thinking_budget: int) -> dict[str, Any]:
    """Build DeepSeek-specific thinking parameters.

    DeepSeek uses:
      {"thinking": {"type": "enabled"}, "reasoning_effort": "high"|"max"}
    ``reasoning_effort`` only distinguishes "high" and "max", so any budget
    below 32768 collapses to "high" — bump the budget to 32768+ for "max".
    """
    if thinking_budget <= 0:
        return {"thinking": {"type": "disabled"}}
    effort = "max" if thinking_budget >= 32768 else "high"
    return {"thinking": {"type": "enabled"}, "reasoning_effort": effort}


def _gemini_thinking_params(thinking_budget: int) -> dict[str, Any]:
    """Build Gemini-specific thinking parameters.

    Gemini's OpenAI-compatible endpoint accepts thinking config wrapped in an
    ``extra_body`` key when making raw JSON requests. Setting
    ``include_thoughts: true`` causes thinking text to be inlined in the
    content wrapped in ``<thought>...</thought>`` tags.
    """
    if thinking_budget <= 0:
        level = "minimal"
    elif thinking_budget <= 2048:
        level = "low"
    elif thinking_budget <= 8192:
        level = "medium"
    else:
        level = "high"
    return {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinking_level": level,
                    "include_thoughts": True,
                }
            }
        }
    }


def _mistral_thinking_params(thinking_budget: int) -> dict[str, Any]:
    """Build Mistral-specific thinking parameters.

    Mistral uses:
      {"reasoning_effort": "high"|"none"}
    """
    if thinking_budget <= 0:
        return {"reasoning_effort": "none"}
    return {"reasoning_effort": "high"}


_THINKING_BUILDERS = {
    "deepseek": _deepseek_thinking_params,
    "gemini": _gemini_thinking_params,
    "mistral": _mistral_thinking_params,
}


def _extract_response(data: dict[str, Any]) -> tuple[str, str]:
    """Extract content and thinking text from a chat completion response.

    Returns ``(content_text, thinking_text)``.  ``thinking_text`` may be empty
    if the model didn't produce reasoning or thinking is disabled.

    Handles provider formats:
    1. DeepSeek: ``message.reasoning_content`` field alongside ``content``
    2. Gemini: thinking is inlined in ``content`` wrapped in
       ``<thought>...</thought>`` tags (when ``include_thoughts=true``)
    3. Array content: ``message.content`` is a list of typed blocks
       ``[{type: "thinking", ...}, {type: "text", ...}]``
    4. Standard: ``message.content`` is a plain string (no thinking)
    """
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    thinking = ""

    # 1. Check for reasoning_content field (DeepSeek)
    reasoning = message.get("reasoning_content", "")
    if reasoning:
        thinking = reasoning

    # 2. Handle array content format (typed blocks)
    if isinstance(content, list):
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking":
                inner = block.get("thinking", {})
                if isinstance(inner, list):
                    for item in inner:
                        if isinstance(item, dict) and item.get("type") == "text":
                            thinking_parts.append(item.get("text", ""))
                elif isinstance(inner, dict):
                    thinking_parts.append(inner.get("text", ""))
                elif isinstance(inner, str):
                    thinking_parts.append(inner)
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            else:
                text_parts.append(block.get("text", ""))
        content_text = "".join(text_parts)
        if thinking_parts:
            thinking = "".join(thinking_parts)
        return content_text, thinking

    # 3. String content — check for Gemini <thought> tags
    if isinstance(content, str):
        content_text, thought_text = _extract_thought_tags(content)
        if thought_text:
            thinking = thought_text
        return content_text, thinking

    return str(content), thinking


def _extract_thought_tags(text: str) -> tuple[str, str]:
    """Extract and strip ``<thought>...</thought>`` blocks from text.

    Returns ``(clean_content, thinking_text)``.
    """
    match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    if not match:
        return text, ""
    thinking = match.group(1).strip()
    clean = text[: match.start()] + text[match.end() :]
    return clean.strip(), thinking


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI-compatible chat completions client using ``requests``.

    Supports thinking/reasoning mode via provider-specific parameters.
    Thinking is **enabled by default** and controlled by ``thinking_budget``.
    """

    def __init__(
        self,
        provider_config: dict[str, Any],
        *,
        thinking_budget: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.base_url = provider_config["base_url"].rstrip("/")
        self.model = provider_config["model"]
        self.api_key = provider_config["api_key"]
        self.supports_json_mode = provider_config.get("supports_json_mode", False)
        self.supports_thinking = provider_config.get("supports_thinking", False)
        self.thinking_style = provider_config.get("thinking_style", "")
        self.provider_name = provider_config.get("name", "Unknown")
        self.verbose = verbose

        # Resolve thinking budget: explicit > preset (set later) > disabled
        self._thinking_budget = thinking_budget if thinking_budget is not None else 0

    @property
    def thinking_budget(self) -> int:
        return self._thinking_budget

    @thinking_budget.setter
    def thinking_budget(self, value: int) -> None:
        self._thinking_budget = max(0, value)

    @property
    def thinking_enabled(self) -> bool:
        return self.supports_thinking and self._thinking_budget > 0

    def _print_thinking(self, thinking_text: str) -> None:
        """Print the model's thinking/reasoning to stderr in verbose mode."""
        # ANSI dim + italic for thinking, reset at end
        dim = "\033[2;3m"
        reset = "\033[0m"
        header = f"\033[2;36m{'─' * 60}\033[0m"

        print(f"\n{header}", file=sys.stderr)
        print(f"\033[2;36m💭 Model thinking:\033[0m", file=sys.stderr)
        print(f"{header}", file=sys.stderr)
        for line in thinking_text.strip().splitlines():
            print(f"{dim}  {line}{reset}", file=sys.stderr)
        print(f"{header}\n", file=sys.stderr)

    # -- core request --------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        json_mode: bool = False,
        max_retries: int = 2,
    ) -> str:
        """Call the chat completions endpoint and return the assistant content.

        Retries on transient HTTP errors (429, 5xx) with exponential backoff.
        Automatically injects thinking parameters when thinking is enabled.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode and self.supports_json_mode:
            body["response_format"] = {"type": "json_object"}

        # Inject thinking parameters
        if self.supports_thinking and self.thinking_style in _THINKING_BUILDERS:
            thinking_params = _THINKING_BUILDERS[self.thinking_style](
                self._thinking_budget
            )
            body.update(thinking_params)

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=body, timeout=(10, 60)
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = requests.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(
                            f"  ⚠ LLM API returned {resp.status_code}, "
                            f"retrying in {wait}s (attempt {attempt}/{max_retries})…",
                            file=sys.stderr,
                        )
                        time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content_text, thinking_text = _extract_response(data)
                if thinking_text and self.verbose:
                    self._print_thinking(thinking_text)
                return content_text
            except requests.RequestException as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"LLM API call failed after {max_retries} attempts: {last_error}"
        )

    # -- convenience methods -------------------------------------------------

    def ask_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_json_retries: int = 1,
    ) -> dict:
        """Send a prompt expecting a JSON response.

        If JSON extraction fails, retries with error feedback appended.
        """
        messages = [{"role": "user", "content": prompt}]
        raw = self.chat_completion(messages, temperature=temperature, json_mode=True)

        try:
            return extract_json(raw)
        except ValueError as first_err:
            if max_json_retries < 1:
                raise
            # Retry with error feedback
            retry_msg = RETRY_JSON_PROMPT.format(error=str(first_err))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": retry_msg})
            print("  ⚠ Invalid JSON from LLM, retrying with feedback…", file=sys.stderr)
            raw2 = self.chat_completion(
                messages, temperature=temperature, json_mode=True
            )
            return extract_json(raw2)



    # -- streaming methods ---------------------------------------------------

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_retries: int = 2,
    ) -> Any:
        """Streaming variant of chat_completion.

        Yields event dicts:
          {"type": "thinking", "delta": "..."}
          {"type": "content", "delta": "..."}
          {"type": "done", "content": "...", "thinking": "..."}

        Uses the OpenAI-compatible streaming API (``stream=True``).  The
        connection is retried on transient errors (429, 5xx) with exponential
        backoff; retries only happen *before* the first chunk is yielded, so
        deltas are never emitted twice.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        # Inject thinking parameters (same as non-streaming)
        if self.supports_thinking and self.thinking_style in _THINKING_BUILDERS:
            thinking_params = _THINKING_BUILDERS[self.thinking_style](
                self._thinking_budget
            )
            body.update(thinking_params)

        # Establish the streaming connection with retries on transient errors.
        resp = None
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=body, timeout=(10, 30), stream=True
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = requests.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                    resp.close()
                    resp = None
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(
                            f"  ⚠ LLM stream returned {last_error}, "
                            f"retrying in {wait}s (attempt {attempt}/{max_retries})…",
                            file=sys.stderr,
                        )
                        time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        if resp is None:
            raise RuntimeError(
                f"LLM streaming call failed after {max_retries} attempts: {last_error}"
            )

        full_content = ""
        full_thinking = ""

        # State machine for Gemini <thought> tags in streaming content.
        # Phases: "detect" (waiting for first content to check for <thought>),
        #         "thinking" (inside <thought> block), "content" (normal content).
        _thought_phase = "detect"
        _thought_buffer = ""  # small buffer to detect the opening tag

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            # SSE format: "data: {...}" or "data: [DONE]"
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = chunk.get("choices", [{}])[0].get("delta", {})

            # 1. Check for reasoning_content (DeepSeek)
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning:
                full_thinking += reasoning
                yield {"type": "thinking", "delta": reasoning}

            # 2. Handle content
            content = delta.get("content")
            if content is None:
                continue

            # Normalise content into a list of typed blocks if applicable.
            blocks: list[dict] | None = None
            if isinstance(content, list):
                blocks = content
            elif isinstance(content, str):
                stripped = content.lstrip()
                if stripped.startswith("["):
                    try:
                        parsed = json.loads(content)
                        if (
                            isinstance(parsed, list)
                            and parsed
                            and isinstance(parsed[0], dict)
                            and "type" in parsed[0]
                        ):
                            blocks = parsed
                    except (json.JSONDecodeError, ValueError):
                        pass

            if blocks is not None:
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "thinking":
                        inner = block.get("thinking", {})
                        if isinstance(inner, list):
                            text = "".join(
                                item.get("text", "")
                                for item in inner
                                if isinstance(item, dict) and item.get("type") == "text"
                            )
                        elif isinstance(inner, dict):
                            text = inner.get("text", "")
                        else:
                            text = str(inner)
                        if text:
                            full_thinking += text
                            yield {"type": "thinking", "delta": text}
                    elif block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            full_content += text
                            yield {"type": "content", "delta": text}
            elif isinstance(content, str) and content:
                # --- Gemini <thought> tag state machine ---
                if _thought_phase == "detect":
                    _thought_buffer += content
                    if "<thought>" in _thought_buffer:
                        # Split at the tag — anything before is content
                        before, _, after = _thought_buffer.partition("<thought>")
                        if before.strip():
                            full_content += before
                            yield {"type": "content", "delta": before}
                        _thought_phase = "thinking"
                        # Process any text after the opening tag
                        if after:
                            if "</thought>" in after:
                                thought_text, _, remainder = after.partition("</thought>")
                                if thought_text:
                                    full_thinking += thought_text
                                    yield {"type": "thinking", "delta": thought_text}
                                _thought_phase = "content"
                                if remainder.strip():
                                    full_content += remainder
                                    yield {"type": "content", "delta": remainder}
                            else:
                                full_thinking += after
                                yield {"type": "thinking", "delta": after}
                    elif len(_thought_buffer) > 20:
                        # No <thought> tag found after enough chars — treat as content
                        _thought_phase = "content"
                        full_content += _thought_buffer
                        yield {"type": "content", "delta": _thought_buffer}
                        _thought_buffer = ""
                elif _thought_phase == "thinking":
                    if "</thought>" in content:
                        thought_text, _, remainder = content.partition("</thought>")
                        if thought_text:
                            full_thinking += thought_text
                            yield {"type": "thinking", "delta": thought_text}
                        _thought_phase = "content"
                        if remainder.strip():
                            full_content += remainder
                            yield {"type": "content", "delta": remainder}
                    else:
                        full_thinking += content
                        yield {"type": "thinking", "delta": content}
                else:
                    # "content" phase — normal content
                    full_content += content
                    yield {"type": "content", "delta": content}

        # Flush any remaining detection buffer
        if _thought_phase == "detect" and _thought_buffer:
            full_content += _thought_buffer
            yield {"type": "content", "delta": _thought_buffer}

        yield {"type": "done", "content": full_content, "thinking": full_thinking}

    def ask_text_stream(
        self, prompt: str, *, temperature: float = 0.4
    ) -> Any:
        """Streaming variant of ask_text. Yields event dicts."""
        messages = [{"role": "user", "content": prompt}]
        yield from self.chat_completion_stream(
            messages, temperature=temperature
        )

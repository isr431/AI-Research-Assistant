"""Search history manager — saves results and maintains a JSON index."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from typing import Any


def _slugify(text: str, max_len: int = 50) -> str:
    """Turn a question into a filesystem-safe slug."""
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "_", slug)
    slug = slug.strip("_")
    return slug[:max_len].rstrip("_")


def generate_search_title(question: str, max_words: int = 7) -> str:
    """Create a short deterministic title from a search question."""
    title = question.strip()
    title = re.sub(r"\s+", " ", title)
    title = title.strip(" \t\r\n?.!,;:")

    # Drop common question openers so sidebar titles scan better.
    title = re.sub(
        r"(?i)^(please\s+)?(can you|could you|would you|tell me|explain|what is|what are|who is|who are|how do|how does|how can|why is|why are|when is|when did|where is)\s+",
        "",
        title,
    )
    title = title.strip(" \t\r\n?.!,;:")

    words = title.split()
    if len(words) > max_words:
        title = " ".join(words[:max_words]).rstrip(" \t\r\n-:;,")

    if not title:
        return "Untitled Search"

    # Preserve all-caps acronyms while giving ordinary questions a title shape.
    return " ".join(
        word if word.isupper() else word[:1].upper() + word[1:]
        for word in title.split()
    )


class SearchHistory:
    """Manages search history as markdown files + a JSON index."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = output_dir
        self._index_path = os.path.join(output_dir, "history.json")
        self._lock = threading.RLock()
        os.makedirs(output_dir, exist_ok=True)

    # -- persistence ----------------------------------------------------------

    def _load_index(self) -> list[dict[str, Any]]:
        if not os.path.exists(self._index_path):
            return []
        with open(self._index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, entries: list[dict[str, Any]]) -> None:
        tmp_path = f"{self._index_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self._index_path)

    # -- public API -----------------------------------------------------------

    def save_search(
        self,
        question: str,
        mode: str,
        provider: str,
        content: str,
        thinking: str = "",
        sources: list[dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> str:
        """Save a search result and return its ID."""
        search_id = uuid.uuid4().hex[:12]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        slug = _slugify(question)
        title = title or generate_search_title(question)
        filename_slug = f"_{slug}" if slug else ""
        filename = f"{timestamp}_{search_id}{filename_slug}.md"
        filepath = os.path.join(self.output_dir, filename)

        with self._lock:
            # Build markdown file while holding the lock so the file and index
            # cannot diverge under concurrent saves/deletes.
            front_matter = (
                f"---\n"
                f"id: {search_id}\n"
                f'title: "{title}"\n'
                f'question: "{question}"\n'
                f"mode: {mode}\n"
                f"provider: {provider}\n"
                f"date: {now.isoformat()}\n"
                f"---\n\n"
            )

            parts = [front_matter, content, "\n"]

            if thinking:
                parts.append("\n---\n\n## Model Thinking\n\n")
                parts.append(thinking)
                parts.append("\n")

            if sources:
                # Sources are already in the response from citation validation,
                # but we also save structured source data as a comment block
                # for programmatic access.
                parts.append("\n<!-- SOURCES_JSON\n")
                parts.append(json.dumps(sources, ensure_ascii=False))
                parts.append("\nSOURCES_JSON -->\n")

            if images:
                parts.append("\n<!-- IMAGES_JSON\n")
                parts.append(json.dumps(images, ensure_ascii=False))
                parts.append("\nIMAGES_JSON -->\n")

            if metadata:
                parts.append("\n<!-- PIPELINE_METADATA_JSON\n")
                parts.append(json.dumps(metadata, ensure_ascii=False))
                parts.append("\nPIPELINE_METADATA_JSON -->\n")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("".join(parts))

            # Update index
            entry = {
                "id": search_id,
                "title": title,
                "question": question,
                "mode": mode,
                "provider": provider,
                "date": now.isoformat(),
                "filename": filename,
                "has_thinking": bool(thinking),
            }
            entries = self._load_index()
            entries.insert(0, entry)  # newest first
            self._save_index(entries)

        return search_id

    def delete_search(self, search_id: str) -> bool:
        """Delete a search result by ID. Returns True if found and deleted."""
        with self._lock:
            entries = self._load_index()
            entry = next((e for e in entries if e["id"] == search_id), None)
            if not entry:
                return False

            # Remove the markdown file
            filepath = os.path.join(self.output_dir, entry["filename"])
            if os.path.exists(filepath):
                os.remove(filepath)

            # Remove from index
            entries = [e for e in entries if e["id"] != search_id]
            self._save_index(entries)
        return True

    def list_searches(self) -> list[dict[str, Any]]:
        """Return all search history entries, newest first."""
        with self._lock:
            return self._load_index()

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        """Load a specific search result by ID."""
        with self._lock:
            entries = self._load_index()
            entry = next((e for e in entries if e["id"] == search_id), None)
            if not entry:
                return None

            filepath = os.path.join(self.output_dir, entry["filename"])
            if not os.path.exists(filepath):
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()

        # Parse content — split at front matter end
        parts = raw.split("---\n", 2)
        if len(parts) >= 3:
            body = parts[2].strip()
        else:
            body = raw

        # Extract thinking section if present
        thinking = ""
        if "\n## Model Thinking\n" in body:
            main_content, _, thinking_section = body.partition(
                "\n## Model Thinking\n"
            )
            # Remove structured JSON comments from thinking
            thinking = re.sub(
                r"\n<!-- (SOURCES_JSON|IMAGES_JSON|PIPELINE_METADATA_JSON)\n.*?\n\1 -->\n?",
                "",
                thinking_section,
                flags=re.DOTALL,
            ).strip()
            # Strip the trailing "---" separator that precedes the thinking
            # section (without chewing off legitimate trailing dashes).
            body = re.sub(r"\n-{3,}\s*$", "", main_content.rstrip()).rstrip()
        else:
            # Remove structured JSON comments from body
            body = re.sub(
                r"\n<!-- (SOURCES_JSON|IMAGES_JSON|PIPELINE_METADATA_JSON)\n.*?\n\1 -->\n?",
                "",
                body,
                flags=re.DOTALL,
            ).strip()

        # Extract structured sources from JSON comment
        sources: list[dict[str, Any]] = []
        sources_match = re.search(
            r"<!-- SOURCES_JSON\n(.*?)\nSOURCES_JSON -->", raw, re.DOTALL
        )
        if sources_match:
            try:
                sources = json.loads(sources_match.group(1))
            except json.JSONDecodeError:
                pass

        images: list[dict[str, Any]] = []
        images_match = re.search(
            r"<!-- IMAGES_JSON\n(.*?)\nIMAGES_JSON -->", raw, re.DOTALL
        )
        if images_match:
            try:
                images = json.loads(images_match.group(1))
            except json.JSONDecodeError:
                pass

        metadata: dict[str, Any] = {}
        metadata_match = re.search(
            r"<!-- PIPELINE_METADATA_JSON\n(.*?)\nPIPELINE_METADATA_JSON -->",
            raw,
            re.DOTALL,
        )
        if metadata_match:
            try:
                metadata = json.loads(metadata_match.group(1))
            except json.JSONDecodeError:
                pass

        return {
            "id": entry["id"],
            "title": entry.get("title") or generate_search_title(entry["question"]),
            "question": entry["question"],
            "mode": entry["mode"],
            "provider": entry["provider"],
            "date": entry["date"],
            "content": body,
            "thinking": thinking,
            "sources": sources,
            "images": images,
            "metadata": metadata,
            "has_thinking": entry.get("has_thinking", False),
        }

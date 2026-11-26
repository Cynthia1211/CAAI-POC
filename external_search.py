import html
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Protocol

from chunk_builder import build_chunks


class ExternalKnowledgeProvider(Protocol):
    """Interface for external knowledge providers."""

    def search(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        ...


def _status_message(found_results: bool, error: str | None) -> str | None:
    """Build a user-facing status message for an external search outcome."""
    if error:
        return f"External source request failed: {error}."
    if not found_results:
        return "No result from external source."
    return None


def _strip_html(raw: str | None) -> str:
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", no_tags).strip()


def _extract_query_terms(query: str) -> str:
    stop = {
        "what", "are", "is", "the", "a", "an", "for", "to", "of", "in", "on",
        "with", "how", "do", "i", "we", "you", "can", "should", "about", "tell",
        "me", "explain", "more", "please",
    }
    tokens = [w for w in re.findall(r"[A-Za-z0-9-]{2,}", query) if w.lower() not in stop]
    return " ".join(tokens[:6]) if tokens else query.strip()


class MedlinePlusTopicProvider:
    """
    Search MedlinePlus health topics and return concise summaries.

    API documentation: https://wsearch.nlm.nih.gov/ws/query
    """

    BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"

    def __init__(self, timeout: float = 6.0, max_summary_chars: int = 500):
        self.timeout = timeout
        self.max_summary_chars = max_summary_chars
        self.last_error: str | None = None

    def search(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        self.last_error = None
        term = _extract_query_terms(query)
        if not term:
            return []

        params = urllib.parse.urlencode({
            "db": "healthTopics",
            "term": " OR ".join(term.split()),
            "retmax": str(max(top_k, 2)),
        })
        url = f"{self.BASE_URL}?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CareAssistantPOC/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
        except Exception as exc:
            self.last_error = type(exc).__name__
            return []

        items: list[dict[str, Any]] = []
        for doc in root.findall(".//document"):
            url_attr = doc.attrib.get("url", "")
            title = ""
            summary = ""

            for content in doc.findall("./content"):
                name = content.attrib.get("name", "")
                text_val = _strip_html("".join(content.itertext()))
                if name == "title" and not title:
                    title = text_val
                elif name in ("FullSummary", "snippet") and text_val:
                    # Prefer the full summary and fall back to the snippet.
                    if name == "FullSummary" or not summary:
                        summary = text_val

            if not title and not summary:
                continue

            if len(summary) > self.max_summary_chars:
                summary = summary[: self.max_summary_chars].rstrip() + "..."

            idx = len(items) + 1
            items.append({
                "chunk": {
                    "id": f"medlineplus_{idx}",
                    "category": "MedlinePlus Health Topic",
                    "section": title or f"MedlinePlus Topic #{idx}",
                    "text": summary or title,
                    "source_name": "MedlinePlus (NIH/NLM)",
                    "url": url_attr or "https://medlineplus.gov/",
                },
                "is_direct": False,
                "via": ["EXTERNAL_API:MedlinePlus"],
                "rank_hint": 0.42,
            })

            if len(items) >= top_k:
                break

        return items


def fetch_external_knowledge(
    query: str,
    top_k: int = 2,
    provider: ExternalKnowledgeProvider | None = None,
) -> dict[str, Any]:
    enabled = os.environ.get("ENABLE_EXTERNAL_SOURCE", "true").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return {
            "searched": False,
            "triggered": False,
            "status_message": None,
            "items": [],
            "chunks": [],
        }

    src = provider or MedlinePlusTopicProvider()
    ext_items = src.search(query=query, top_k=top_k)
    error = getattr(src, "last_error", None)
    triggered = bool(ext_items)
    status_message = _status_message(triggered, error)
    query_results = []
    detail_values = []
    for item in ext_items:
        source_chunk = item.get("chunk", {})
        url = source_chunk.get("url", "")
        title = source_chunk.get("section", "")
        text = source_chunk.get("text", "")
        query_results.append({
            "source": f"External Source：{url}",
            "records": [{"fields": {"title": title, "text": text}}],
        })
        detail_values.append(f"{title}\n{text}".strip())

    chunks = build_chunks(
        query_results,
        chunk_prefix="external",
        empty_detail="No external result returned.",
    )
    for chunk, detail in zip(chunks, detail_values):
        chunk["Detail"] = detail

    return {
        "searched": True,
        "triggered": triggered,
        "status_message": status_message,
        "items": ext_items,
        "chunks": chunks,
    }

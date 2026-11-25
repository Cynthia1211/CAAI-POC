"""Grounded context builder for SQL and Neptune result chunks."""

from typing import Any


def build_grounded_context_new(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Join structured SQL, Graph, and hybrid-search chunks into grounded context."""
    blocks: list[str] = []
    normalized: list[dict[str, Any]] = []

    for chunk in chunks:
        item = dict(chunk)
        normalized.append(item)
        lines = [
            f"Chunk Name: {item['Chunk Name']}",
            f"Source: {item['Source']}",
        ]
        for field in ("Category", "Section", "Matched"):
            if item.get(field):
                lines.append(f"{field}: {item[field]}")
        if "Detail" in item:
            lines.append(f"Detail: {item['Detail']}")
        elif "Details" in item:
            lines.append(f"Details: {item['Details']}")
        if "Entities" in item:
            lines.append(f"Entities: {item['Entities']}")
        blocks.append("\n".join(lines))

    return {
        "count": len(normalized),
        "chunks": normalized,
        "context_text": "\n\n---\n\n".join(blocks)
        if blocks
        else "No records matched the question.",
    }

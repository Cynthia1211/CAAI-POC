"""Placeholder calendar and reminder tool returning the shared chunk shape."""

from chunk_builder import build_chunks


def run_calendar_reminder(query: str = "") -> list[dict]:
    """Return one placeholder chunk until calendar/reminder lookup is implemented."""
    chunks = build_chunks(
        [{
            "source": "calendar&reminder",
            "records": [{"fields": {"content": "Calendar/reminder tool chunk"}}],
        }],
        chunk_prefix="calendar&reminder",
        empty_detail="Calendar/reminder tool chunk",
        start_index=0,
    )
    if chunks:
        chunks[0]["Detail"] = "Calendar/reminder tool chunk"
    return chunks

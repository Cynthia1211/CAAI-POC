"""Service entry point for running the LangGraph question workflow."""

from collections.abc import Callable
from typing import Any

from rag_graph import run_rag_graph


def handle_question(
    query: str,
    current_user: str = "USER-001",
    progress_callback: Callable[[str], None] | None = None,
    session_id: str = "default-session",
) -> dict[str, Any]:
    """Run a question through the graph and return the API response payload."""
    return run_rag_graph(
        query=query,
        current_user=current_user,
        progress_callback=progress_callback,
        session_id=session_id,
    )

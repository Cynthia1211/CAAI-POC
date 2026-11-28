"""Calendar subgraph; the parent graph owns the conversation checkpoint."""

from typing import Any, Callable, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from calendar_utils import (
    CalendarOperation, CalendarProvider, extract_calendar_operation,
    _run_calendar_create, _run_calendar_delete, _run_calendar_view,
)


class CalendarGraphState(TypedDict, total=False):
    query: str
    history: list[dict[str, str]]
    provider: CalendarProvider
    pending_delete_ids: list[str]
    operation: CalendarOperation | None
    calendar_chunks: list[dict[str, Any]]
    trace: list[str]


def extract_node(state: CalendarGraphState, config: RunnableConfig) -> dict:
    operation = extract_calendar_operation(
        state["query"], history=state["history"],
        pending_ids=state["pending_delete_ids"],
    )
    message = f"Running calendar {operation.action} operation..."
    callback = (config.get("configurable") or {}).get("progress_callback")
    if callback:
        callback(message)
    return {"operation": operation, "trace": [message]}


def view_node(state: CalendarGraphState) -> dict:
    chunks = _run_calendar_view(state["operation"], state["provider"])
    return {"calendar_chunks": chunks, "pending_delete_ids": []}


def create_node(state: CalendarGraphState) -> dict:
    chunks = _run_calendar_create(state["operation"], state["provider"])
    return {"calendar_chunks": chunks, "pending_delete_ids": []}


def delete_node(state: CalendarGraphState) -> dict:
    chunks = _run_calendar_delete(state["operation"], state["provider"], state["pending_delete_ids"])
    return {"calendar_chunks": chunks, "pending_delete_ids": chunks[0].get("pending_delete_ids", [])}


def _build_calendar_subgraph():
    graph = StateGraph(CalendarGraphState)
    graph.add_node("extract_node", extract_node)
    graph.add_node("view_node", view_node)
    graph.add_node("create_node", create_node)
    graph.add_node("delete_node", delete_node)

    graph.add_edge(START, "extract_node")
    graph.add_conditional_edges(
        "extract_node", lambda state: state["operation"].action,
        {"view": "view_node", "create": "create_node", "delete": "delete_node"},
    )
    graph.add_edge("view_node", END)
    graph.add_edge("create_node", END)
    graph.add_edge("delete_node", END)
    return graph.compile(checkpointer=False)


_calendar_subgraph = _build_calendar_subgraph()


def run_calendar_subgraph(
    query: str, *, provider: CalendarProvider, history: list[dict[str, str]] | None = None,
    pending_delete_ids: list[str] | None = None, progress_callback: Callable[[str], None] | None = None,
) -> dict:
    state: CalendarGraphState = dict(
        query=query, provider=provider, history=history or [],
        pending_delete_ids=pending_delete_ids or [], trace=[],
    )
    result = _calendar_subgraph.invoke(state, config={"configurable": {"progress_callback": progress_callback}})
    chunks = result["calendar_chunks"]
    message = f"Calendar operation produced {len(chunks)} chunk(s)."
    if progress_callback:
        progress_callback(message)
    return {
        "calendar_chunks": chunks,
        "pending_delete_ids": result["pending_delete_ids"],
        "trace": [*result["trace"], message],
    }


# Calendar subgraph:
# python -c 'from calendar_graph import _calendar_subgraph; print(_calendar_subgraph.get_graph().draw_mermaid())'

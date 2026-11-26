"""LangGraph workflow: retrieve from the available sources, ground, and answer."""

from collections.abc import Callable
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from answer_generator import answer_from_grounded_context, answer_general_question
from graph_search import extract_entities_from_chunks, search_neptune_relationships
from grounded_context_new import build_grounded_context_new
from hybrid_search import hybrid_search
from intent_router import classify_intent
from postgreSQL_search import run_structured_query
from external_search import fetch_external_knowledge
from calendar_reminder import (
    add_calendar_event,
    calendar_events_to_chunks,
    calendar_updated_to_chunk,
    delete_calendar_event,
    view_calendar_events,
)


class QuestionState(TypedDict, total=False):
    query: str
    current_user: str
    intent: dict[str, Any]
    sql_chunks: list[dict[str, Any]]
    graph_chunks: list[dict[str, Any]]
    hybrid_chunks: list[dict[str, Any]]
    external_chunks: list[dict[str, Any]]
    calendar_chunks: list[dict[str, Any]]
    entities: list[dict[str, str]]
    grounded_context: dict[str, Any]
    answer_result: dict[str, Any]
    trace: list[str]


def _build_rag_graph(progress_callback: Callable[[str], None] | None = None):
    """Build a request-scoped graph so progress reporting stays request-scoped too."""
    def log(state: QuestionState, message: str) -> list[str]:
        if progress_callback:
            progress_callback(message)
        return [*state.get("trace", []), message]

    def intent_node(state: QuestionState) -> dict:
        trace = log(state, "Classifying question intent...")
        classification = classify_intent(state["query"])
        trace = log(
            {**state, "trace": trace},
            f"Classified intent: {classification.intent}",
        )
        return {"intent": classification.model_dump(), "trace": trace}

    def structured_node(state: QuestionState) -> dict:
        trace = log(state, "Running PostgreSQL search...")
        chunks = run_structured_query(state["query"], current_user=state["current_user"])
        trace = log({**state, "trace": trace}, f"PostgreSQL returned {len(chunks)} query chunk(s).")
        return {"sql_chunks": chunks, "entities": extract_entities_from_chunks(chunks), "trace": trace}

    def graph_node(state: QuestionState) -> dict:
        trace = log(state, "Searching Neptune relationships...")
        chunks = search_neptune_relationships(state.get("sql_chunks", []))
        trace = log({**state, "trace": trace}, f"Graph lookup returned {len(chunks)} table-group chunk(s).")
        return {"graph_chunks": chunks, "trace": trace}

    def hybrid_node(state: QuestionState) -> dict:
        intent = state["intent"]["intent"]
        if intent == "personal":
            table_name = "personal_chunks"
            chunk_user_id = state["current_user"]
        else:
            # Health knowledge chunks are shared and stored with a NULL user_id.
            table_name = "doc_chunks"
            chunk_user_id = None

        trace = log(
            state,
            f"Running hybrid document search in {table_name} (user_id={chunk_user_id})...",
        )
        chunks = hybrid_search(
            state["query"],
            table_name=table_name,
            user_id=chunk_user_id,
        )
        trace = log({**state, "trace": trace}, f"Hybrid search returned {len(chunks)} chunk(s).")
        return {"hybrid_chunks": chunks, "trace": trace}

    def context_node(state: QuestionState) -> dict:
        trace = log(state, "Building grounded context...")
        chunks = (
            state.get("sql_chunks", [])
            + state.get("graph_chunks", [])
            + state.get("hybrid_chunks", [])
            + state.get("external_chunks", [])
            + state.get("calendar_chunks", [])
        )
        context = build_grounded_context_new(chunks)
        return {"grounded_context": context, "trace": trace}

    def answer_node(state: QuestionState) -> dict:
        trace = log(state, "Generating answer...")
        result = answer_from_grounded_context(state["query"], state["grounded_context"])
        return {"answer_result": result, "trace": trace}

    def general_answer_node(state: QuestionState) -> dict:
        trace = log(state, "Answering general question directly...")
        result = answer_general_question(state["query"])
        return {"answer_result": result, "trace": trace}

    def calendar_reminder_node(state: QuestionState) -> dict:
        classification = state["intent"]
        action = classification.get("calendar_action") or "view"
        trace = log(state, f"Running calendar {action} operation...")
        if action == "view":
            date_from = classification.get("date_from")
            date_to = classification.get("date_to")
            time_zone = classification.get("time_zone") or "America/Toronto"
            events = view_calendar_events(
                date_from=date_from,
                date_to=date_to,
                time_zone=time_zone,
            )
            window_note = (
                f"Reference date: {date.today().isoformat()}. "
                f"Calendar search window: {date_from or 'now'} through "
                f"{date_to or 'no end date'} ({time_zone})."
            )
            chunks = calendar_events_to_chunks(events, result_note=window_note)
        elif action == "create":
            summary = classification.get("summary")
            start_iso = classification.get("start_iso")
            end_iso = classification.get("end_iso")
            time_zone = classification.get("time_zone") or "America/Toronto"
            if not summary or not start_iso or not end_iso:
                chunks = calendar_events_to_chunks(
                    [], result_note="Please provide an event title, date, start time, and end time."
                )
            else:
                create_result = add_calendar_event(
                    summary,
                    start_iso,
                    end_iso,
                    time_zone,
                    interpret_as_local_time=True,
                )
                chunks = [calendar_updated_to_chunk(create_result)]
        else:
            events = view_calendar_events()
            if not events:
                chunks = calendar_events_to_chunks(
                    [], result_note="There are no upcoming events to delete."
                )
            else:
                event_to_delete = events[0]
                delete_calendar_event(event_to_delete["id"])
                chunks = calendar_events_to_chunks(
                    [event_to_delete], result_note="Event deleted successfully."
                )
        trace = log(
            {**state, "trace": trace},
            f"Calendar operation produced {len(chunks)} chunk(s).",
        )
        return {"calendar_chunks": chunks, "trace": trace}

    def external_node(state: QuestionState) -> dict:
        trace = log(state, "Searching external health sources...")
        result = fetch_external_knowledge(state["query"])
        chunks = result.get("chunks", [])
        trace = log({**state, "trace": trace}, f"External health search returned {len(chunks)} result(s).")
        return {"external_chunks": chunks, "trace": trace}

    graph = StateGraph(QuestionState)
    graph.add_node("intent_router", intent_node)
    graph.add_node("postgresql", structured_node)
    graph.add_node("graph_rag", graph_node)
    graph.add_node("hybrid_rag", hybrid_node)
    graph.add_node("external_search", external_node)
    graph.add_node("calendar_reminder", calendar_reminder_node)
    graph.add_node("ground_context", context_node)
    graph.add_node("answer", answer_node)
    graph.add_node("general_answer", general_answer_node)
    graph.add_edge(START, "intent_router")
    def route_intent(state: QuestionState) -> str:
        return state["intent"]["intent"]

    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "personal": "postgresql",
            "calendar": "calendar_reminder",
            "health": "hybrid_rag",
            "general": "general_answer",
        },
    )
    def route_after_hybrid(state: QuestionState) -> str:
        return state["intent"]["intent"]

    graph.add_edge("postgresql", "graph_rag")
    graph.add_edge("graph_rag", "hybrid_rag")
    graph.add_conditional_edges(
        "hybrid_rag",
        route_after_hybrid,
        {
            "personal": "ground_context",
            "health": "external_search",
        },
    )
    graph.add_edge("external_search", "ground_context")
    graph.add_edge("calendar_reminder", "ground_context")
    graph.add_edge("ground_context", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("general_answer", END)
    return graph.compile()


def run_rag_graph(
    query: str,
    current_user: str = "USER-001",
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run the available retrieval tools through LangGraph and return the API payload."""
    state = _build_rag_graph(progress_callback).invoke({
        "query": query,
        "current_user": current_user,
        "trace": [],
    })
    intent = state["intent"]["intent"]
    context = state.get("grounded_context", {
        "count": 0,
        "chunks": [],
        "context_text": "",
    })
    answer = state["answer_result"]
    return {
        "query": query,
        "route": intent,
        "reason": state["intent"]["reason"],
        "intent": state["intent"],
        "trace": state["trace"],
        "result": {
            "answer": answer["answer"],
            "entities": state.get("entities", []),
            "total_chunks": context["count"],
            "chunks": context["chunks"],
            "grounded_context": context["context_text"],
            "sources": answer["sources"],
        },
    }

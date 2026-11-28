"""LangGraph workflow: retrieve from the available sources, ground, and answer."""

from collections.abc import Callable
from typing import Any, TypedDict, Annotated

from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from answer_generator import answer_from_grounded_context, answer_general_question
from chunk_builder import build_chunks
from graph_search import extract_entities_from_chunks, search_user_graph
from grounded_context_new import build_grounded_context_new
from hybrid_search import hybrid_search
from intent_router import classify_intent
from postgreSQL_search import get_records_by_ids
from external_search import fetch_external_knowledge
import calendar_google
from calendar_graph import run_calendar_subgraph


class QuestionState(TypedDict, total=False):
    query: str
    current_user: str
    intent: dict[str, Any]
    sql_chunks: list[dict[str, Any]]
    graph_chunks: list[dict[str, Any]]
    candidate_personal_chunk_ids: list[str]
    hybrid_chunks: list[dict[str, Any]]
    external_chunks: list[dict[str, Any]]
    calendar_chunks: list[dict[str, Any]]
    pending_delete_ids: list[str]
    entities: list[dict[str, str]]
    grounded_context: dict[str, Any]
    answer_result: dict[str, Any]
    trace: list[str]
    messages: Annotated[list[AnyMessage], add_messages]


def _reply_update(query: str, result: dict[str, Any]) -> QuestionState:
    return {
        "answer_result": result,
        "messages": [HumanMessage(content=query), AIMessage(content=result.get("answer", ""))],
    }


def _build_rag_graph(checkpointer: Any | None = None):
    """Build and compile the graph with an injectable checkpoint backend."""
    def recent_history(state: QuestionState) -> list[dict[str, str]]:
        return [
            {
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": str(message.content),
            }
            for message in state.get("messages", [])[-8:]
        ]

    def log(state: QuestionState, message: str, config: RunnableConfig) -> list[str]:
        callback = (config.get("configurable") or {}).get("progress_callback")
        if callback:
            callback(message)
        return [*state.get("trace", []), message]

    def intent_node(state: QuestionState, config: RunnableConfig) -> dict:
        trace = log(state, "Classifying question intent...", config)
        classification = classify_intent(state["query"], history=recent_history(state))
        trace = log(
            {**state, "trace": trace},
            f"Classified intent: {classification.intent}",
            config,
        )
        result = {"intent": classification.model_dump(), "trace": trace}
        if classification.intent != "calendar":
            result["pending_delete_ids"] = []
        return result

    def structured_node(state: QuestionState, config: RunnableConfig) -> dict:
        trace = log(state, "Running PostgreSQL search...", config)
        profile_records = get_records_by_ids(
            "user_profiles",
            [state["current_user"]],
            id_column="user_id",
        )
        chunks = build_chunks(
            [profile_records] if profile_records else [],
            chunk_prefix="PostgreSQL",
            empty_detail="No rows returned.",
            entity_fields={"user_profiles": "user_id"},
        )
        trace = log({**state, "trace": trace}, f"PostgreSQL returned {len(chunks)} query chunk(s).", config)
        return {
            "sql_chunks": chunks,
            "entities": extract_entities_from_chunks(chunks),
            "trace": trace,
        }

    def graph_node(state: QuestionState, config: RunnableConfig) -> dict:
        trace = log(state, "Searching Neptune relationships...", config)
        result = search_user_graph(state["current_user"])
        chunks = result["chunks"]
        entities = state.get("entities", []) + extract_entities_from_chunks(chunks)
        deduplicated_entities = []
        seen_entities = set()
        for entity in entities:
            key = tuple(entity.items())
            if key not in seen_entities:
                seen_entities.add(key)
                deduplicated_entities.append(entity)
        trace = log({**state, "trace": trace}, f"Graph lookup returned {len(chunks)} table-group chunk(s).", config)
        return {
            "graph_chunks": chunks,
            "entities": deduplicated_entities,
            "candidate_personal_chunk_ids": result["personal_chunk_ids"],
            "trace": trace,
        }

    def hybrid_node(state: QuestionState, config: RunnableConfig) -> dict:
        intent = state["intent"]["intent"]
        if intent == "personal":
            table_name = "personal_chunks"
        else:
            # Health knowledge chunks are shared and stored without a user_id column.
            table_name = "doc_chunks"

        trace = log(state, f"Running hybrid document search in {table_name}...", config)
        chunks = hybrid_search(
            state["query"],
            table_name=table_name,
            candidate_ids=(
                state.get("candidate_personal_chunk_ids", [])
                if table_name == "personal_chunks"
                else None
            ),
        )
        trace = log({**state, "trace": trace}, f"Hybrid search returned {len(chunks)} chunk(s).", config)
        return {"hybrid_chunks": chunks, "trace": trace}

    def context_node(state: QuestionState, config: RunnableConfig) -> dict:
        trace = log(state, "Building grounded context...", config)
        chunks = (
            state.get("sql_chunks", [])
            + state.get("graph_chunks", [])
            + state.get("hybrid_chunks", [])
            + state.get("external_chunks", [])
            + state.get("calendar_chunks", [])
        )
        context = build_grounded_context_new(chunks)
        return {"grounded_context": context, "trace": trace}

    def answer_node(state: QuestionState, config: RunnableConfig) -> dict:
        trace = log(state, "Generating answer...", config)
        chunks = state.get("calendar_chunks", []) if state["intent"]["intent"] == "calendar" else []
        if chunks and "delete_receipt" in chunks[0]:
            chunk = chunks[0]
            result = {
                "answer": f"{chunk['delete_receipt']}\n\nSources used:\n- Chunk Name: {chunk['Chunk Name']}\n  Source: {chunk['Source']}",
                "sources": [chunk],
                "source_chunk_names": [chunk["Chunk Name"]],
            }
        else:
            result = answer_from_grounded_context(
                state["query"], state["grounded_context"], history=recent_history(state)
            )
        return {**_reply_update(state["query"], result), "trace": trace}

    def general_answer_node(state: QuestionState, config: RunnableConfig) -> dict:
        trace = log(state, "Answering general question directly...", config)
        history = recent_history(state)
        result = answer_general_question(state["query"], history=history)
        return {**_reply_update(state["query"], result), "trace": trace}

    def calendar_node(state: QuestionState, config: RunnableConfig) -> dict:
        result = run_calendar_subgraph(
            state["query"], provider=calendar_google, history=recent_history(state),
            pending_delete_ids=state.get("pending_delete_ids", []),
            progress_callback=(config.get("configurable") or {}).get("progress_callback"),
        )
        return {**result, "trace": [*state.get("trace", []), *result["trace"]]}

    def external_node(state: QuestionState, config: RunnableConfig) -> dict:
        trace = log(state, "Searching external health sources...", config)
        result = fetch_external_knowledge(state["query"])
        chunks = result.get("chunks", [])
        trace = log({**state, "trace": trace}, f"External health search returned {len(chunks)} result(s).", config)
        return {"external_chunks": chunks, "trace": trace}

    graph = StateGraph(QuestionState)
    graph.add_node("intent_router", intent_node)
    graph.add_node("postgresql", structured_node)
    graph.add_node("graph_rag", graph_node)
    graph.add_node("hybrid_rag", hybrid_node)
    graph.add_node("external_search", external_node)
    graph.add_node("calendar", calendar_node)
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
            "calendar": "calendar",
            "health": "hybrid_rag",
            "general": "general_answer",
        },
    )
    graph.add_edge("postgresql", "graph_rag")
    graph.add_edge("graph_rag", "hybrid_rag")
    graph.add_conditional_edges(
        "hybrid_rag",
        route_intent,
        {
            "personal": "ground_context",
            "health": "external_search",
        },
    )
    graph.add_edge("external_search", "ground_context")
    graph.add_edge("calendar", "ground_context")
    graph.add_edge("ground_context", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("general_answer", END)
    return graph.compile(checkpointer=checkpointer)


# Keep the backend swappable while using in-memory session checkpoints locally.
_checkpointer = MemorySaver()
_rag_app = _build_rag_graph(checkpointer=_checkpointer)


def run_rag_graph(
    query: str,
    current_user: str = "USER-001",
    progress_callback: Callable[[str], None] | None = None,
    session_id: str = "default-session",
) -> dict:
    """Run the available retrieval tools through LangGraph and return the API payload."""
    state = _rag_app.invoke({
        "query": query,
        "current_user": current_user,
        "trace": [],
    }, config={"configurable": {"thread_id": session_id, "progress_callback": progress_callback}})
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
            "answer": answer.get("answer", ""),
            "entities": state.get("entities", []),
            "total_chunks": context["count"],
            "chunks": context["chunks"],
            "grounded_context": context["context_text"],
            "sources": answer["sources"],
        },
    }


# command to get Mermaid code for https://mermaid.ai/app/dashboard
# python -c 'from rag_graph import _rag_app; print(_rag_app.get_graph().draw_mermaid())'

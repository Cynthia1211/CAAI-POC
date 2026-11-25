"""Question workflow: search PostgreSQL, Neptune, and documents, then answer."""

from collections.abc import Callable

from answer_generator import answer_from_grounded_context
from graph_search import extract_entities_from_chunks, search_neptune_relationships
from grounded_context_new import build_grounded_context_new
from hybrid_search import hybrid_search
from structured_query import run_structured_query


def handle_question(
    query: str,
    current_user: str = "USER-001",
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run SQL, graph, and document retrieval for one user question."""
    trace: list[str] = []

    def log(message: str) -> None:
        trace.append(message)
        print(message, flush=True)
        if progress_callback:
            progress_callback(message)

    log("[1/5] Running structured SQL search...")
    sql_chunks = run_structured_query(query, current_user=current_user)

    entities = extract_entities_from_chunks(sql_chunks)
    log(f"PostgreSQL returned {len(sql_chunks)} query chunk(s).")
    log(f"Entities: {entities}")
    log("Structured query chunks:")
    for chunk in sql_chunks:
        log(f"  {chunk['Chunk Name']} | {chunk['Source']}")
    if not sql_chunks:
        log("  None")

    log("[2/5] Searching Neptune relationships...")
    graph_chunks = search_neptune_relationships(sql_chunks)
    log(f"Graph lookup returned {len(graph_chunks)} table-group chunk(s).")
    log("Graph search chunks:")
    for chunk in graph_chunks:
        log(f"  {chunk['Chunk Name']} | {chunk['Source']}")
    if not graph_chunks:
        log("  None")

    log("[3/5] Running hybrid document search...")
    hybrid_chunks = hybrid_search(query, user_id=current_user)
    log(f"Hybrid search returned {len(hybrid_chunks)} chunk(s).")
    log("Hybrid search chunks:")
    for chunk in hybrid_chunks:
        log(f"  {chunk['Chunk Name']} | {chunk['Source']}")
    if not hybrid_chunks:
        log("  None")

    log("[4/5] Building grounded context...")
    chunks = sql_chunks + graph_chunks + hybrid_chunks
    all_entities: list[dict[str, str]] = []
    seen_entities: set[tuple[str, str]] = set()
    for entity in entities:
        for entity_type, entity_id in entity.items():
            key = (entity_type, entity_id)
            if key not in seen_entities:
                seen_entities.add(key)
                all_entities.append({entity_type: entity_id})

    grounded_context = build_grounded_context_new(chunks)

    log("[5/5] Generating answer...")
    answer_result = answer_from_grounded_context(
        query=query,
        grounded_context=grounded_context,
    )
    log(f"User query: {query}")
    log("Generated Answer:")
    log(answer_result["answer"])

    result = {
        "answer": answer_result["answer"],
        "entities": all_entities,
        "total_chunks": len(grounded_context["chunks"]),
        "chunks": grounded_context["chunks"],
        "grounded_context": grounded_context["context_text"],
        "sources": answer_result["sources"],
    }
    return {
        "query": query,
        "route": "structured_retrieval",
        "reason": "LLM query extraction followed by PostgreSQL, Neptune, and hybrid document retrieval.",
        "trace": trace,
        "result": result,
    }

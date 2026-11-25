"""Inspect one-hop graph entities and their grouped PostgreSQL record chunks."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunk_builder import build_chunks
from graph_search import (
    extract_entities_from_chunks,
    find_one_hop_entities,
    group_entities_by_table,
    query_records_for_entity_groups,
)
from structured_query import extract_structured_query, query_postgres_tables


QUERY = "show my family and appointment"
USER_ID = "USER-001"


def main() -> None:
    # The structured result is input only; its SQL/query-plan details are not printed.
    query_spec = extract_structured_query(QUERY)
    sql_chunks = query_postgres_tables(query_spec, current_user=USER_ID)
    received_entities = extract_entities_from_chunks(sql_chunks)

    print("Entities received by graph search:")
    print(f"User ID: {USER_ID}")
    print(received_entities)

    graph_result = find_one_hop_entities(received_entities)

    print("\nOne-hop entities grouped by SQL table:")
    grouped_entities = group_entities_by_table(graph_result["entities"])
    for group in grouped_entities:
        print(f"  {group['table']}: {group['entities']}")

    record_groups = query_records_for_entity_groups(
        grouped_entities,
        relations=graph_result["relations"],
    )

    print("\nRDS records retrieved for each entity group:")
    if not record_groups:
        print("  No entity groups were queried.")
    for group in record_groups:
        print(f"  Table: {group['source']}")
        print(f"  Records: {group['records']}")
        print(f"  Graph relations: {group['graph_relations']}")

    graph_chunks = build_chunks(
        record_groups,
        chunk_prefix="Graph",
        empty_detail="No related rows returned.",
    )

    print("\nGraph chunks by table group:")
    for chunk in graph_chunks:
        print(f"  Chunk Name: {chunk['Chunk Name']}")
        print(f"  Source: {chunk['Source']}")
        print(f"  Detail: \n{chunk['Detail']}")
        print(f"  Entities: {chunk['Entities']}")
        print()


if __name__ == "__main__":
    main()

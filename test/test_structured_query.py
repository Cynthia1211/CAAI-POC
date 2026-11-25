"""Print the Bedrock query extraction, generated SQL, and PostgreSQL results."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from structured_query import (
    build_postgres_queries,
    extract_structured_query,
    query_postgres_tables,
)
from graph_search import extract_entities_from_chunks


QUERY = "show my family and appointment"
USER_ID = "USER-001"


def main() -> None:
    spec = extract_structured_query(QUERY)
    sql_queries = build_postgres_queries(spec, current_user=USER_ID)
    chunks = query_postgres_tables(spec, current_user=USER_ID)
    entities = extract_entities_from_chunks(chunks)

    print("=" * 72)
    print(f"Question: {QUERY}")
    print(f"User ID: {USER_ID}")
    print("LLM extracted fields:")
    for field, value in spec.model_dump().items():
        print(f"  {field}: {value}")

    if len(chunks) != len(sql_queries):
        raise AssertionError("Expected exactly one PostgreSQL chunk per SQL query.")

    print("\nGenerated SQL queries:")
    print(f"SQL query count: {len(sql_queries)}")
    for index, statement in enumerate(sql_queries, start=1):
        print(f"{index}. {statement['query']}")
        print(f"   Parameters: {statement['params']}")

    print("\nPostgreSQL chunks:")
    print(f"Chunk count: {len(chunks)}")
    if not chunks:
        print("  No SQL queries were executed.")
    print(f"Entities: {entities}")
    print("=" * 72)
    for chunk in chunks:
        print(f"  Chunk Name: {chunk['Chunk Name']}")
        print(f"  Source: {chunk['Source']}")
        print(f"  Detail:\n{chunk['Detail']}")
        print(f"  Entities: {chunk['Entities']}")
        print("-" * 72)
    print("=" * 72)


if __name__ == "__main__":
    main()

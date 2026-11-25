"""Build and print grounded context from SQL, graph, and hybrid search."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from graph_search import search_neptune_relationships
from grounded_context_new import build_grounded_context_new
from hybrid_search import hybrid_search
from structured_query import run_structured_query


QUERY = "show my family and appointment"
USER_ID = "USER-001"


def main() -> None:
    sql_chunks = run_structured_query(QUERY, current_user=USER_ID)
    graph_chunks = search_neptune_relationships(sql_chunks)
    hybrid_chunks = hybrid_search(QUERY, user_id=USER_ID)
    chunks = sql_chunks + graph_chunks + hybrid_chunks
    grounded = build_grounded_context_new(chunks)

    print("=" * 72)
    print(f"User query: {QUERY}")
    print(f"User ID: {USER_ID}")
    print("Chunks:")
    for chunk in chunks:
        print(f"  {chunk['Chunk Name']} | {chunk['Source']}")
    print("\nComplete grounded context:")
    print("-" * 72)
    print(grounded["context_text"])
    print("=" * 72)


if __name__ == "__main__":
    main()

"""Build and print grounded context from SQL, graph, and hybrid search."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from chunk_builder import build_chunks
from graph_search import search_user_graph
from grounded_context_new import build_grounded_context_new
from hybrid_search import hybrid_search
from postgreSQL_search import get_records_by_ids


QUERY = "show my family and appointment"
USER_ID = "USER-001"
TABLE_NAME = "personal_chunks"


def main() -> None:
    profile_records = get_records_by_ids("user_profiles", [USER_ID], id_column="user_id")
    sql_chunks = build_chunks(
        [profile_records] if profile_records else [],
        chunk_prefix="PostgreSQL",
        empty_detail="No rows returned.",
        entity_fields={"user_profiles": "user_id"},
    )
    graph_result = search_user_graph(USER_ID)
    graph_chunks = graph_result["chunks"]
    hybrid_chunks = hybrid_search(
        QUERY,
        table_name=TABLE_NAME,
        candidate_ids=graph_result["personal_chunk_ids"],
    )
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

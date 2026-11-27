"""Inspect three-hop graph retrieval and PostgreSQL-hydrated graph chunks."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graph_search import search_user_graph


USER_ID = "USER-001"


def main() -> None:
    result = search_user_graph(USER_ID)
    print(f"User ID: {USER_ID}")
    print(f"Personal chunk candidates: {result['personal_chunk_ids']}")
    for chunk in result["chunks"]:
        print(f"\n{chunk['Chunk Name']} ({chunk['Source']})")
        print(chunk["Detail"])
        print(f"Entities: {chunk['Entities']}")


if __name__ == "__main__":
    main()

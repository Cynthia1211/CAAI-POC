"""Hybrid-search a selected chunk table or an explicit set of chunk IDs."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from hybrid_search import hybrid_search


QUERY = "What do I like?"
TABLE_NAME = "personal_chunks"  # Choose "personal_chunks" or "doc_chunks".
CANDIDATE_IDS: list[str] | None = ['chunk_01', 'chunk_02']  # Use None to search the full table.


def main() -> None:
    # if TABLE_NAME not in {"personal_chunks", "doc_chunks"}:
    #     raise ValueError("TABLE_NAME must be 'personal_chunks' or 'doc_chunks'.")
    # if TABLE_NAME == "personal_chunks" and not CANDIDATE_IDS:
    #     print("Set CANDIDATE_IDS to graph-owned IDs before searching personal_chunks.")
    #     return

    chunks = hybrid_search(
        QUERY,
        table_name=TABLE_NAME,
        top_k=3,
        candidate_ids=CANDIDATE_IDS,
    )

    print("=" * 72)
    print(f"User query: {QUERY}")
    print(f"Table: {TABLE_NAME}")
    print(f"Candidate IDs: {CANDIDATE_IDS if CANDIDATE_IDS is not None else 'all rows'}")
    print(f"Returned chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        print("-" * 72)
        print(f"{index}. Chunk Name: {chunk['Chunk Name']}")
        print(f"Source: {chunk['Source']}")
        # print(f"Detail: {chunk['Detail']}")
    print("=" * 72)


if __name__ == "__main__":
    main()

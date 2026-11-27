"""Inspect current-user PostgreSQL profile retrieval."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunk_builder import build_chunks
from postgreSQL_search import get_records_by_ids


USER_ID = "USER-002"


def main() -> None:
    records = get_records_by_ids("user_profiles", [USER_ID], id_column="user_id")
    chunks = build_chunks(
        [records] if records else [],
        chunk_prefix="PostgreSQL",
        empty_detail="No rows returned.",
        entity_fields={"user_profiles": "user_id"},
    )
    print(f"User ID: {USER_ID}")
    print(f"Profile chunk count: {len(chunks)}")
    for chunk in chunks:
        print(f"{chunk['Chunk Name']} ({chunk['Source']})")
        print(chunk["Detail"])
        print(f"Entities: {chunk['Entities']}")


if __name__ == "__main__":
    main()

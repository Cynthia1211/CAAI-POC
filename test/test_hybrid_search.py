"""Call hybrid_search with a user query and print the returned chunks."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from hybrid_search import hybrid_search


QUERY = "Who usually takes me to my Routine clinic follow-up?"
USER_ID = "USER-002"


def main() -> None:
    chunks = hybrid_search(QUERY, user_id=USER_ID)

    print("=" * 72)
    print(f"User query: {QUERY}")
    print(f"User ID: {USER_ID}")
    print(f"Returned chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        print("-" * 72)
        print(f"{index}. Chunk Name: {chunk['Chunk Name']}")
        print(f"Category: {chunk['Category']}")
        print(f"Section: {chunk['Section']}")
        print(f"Source: {chunk['Source']}")
        print(f"matched: {chunk['Matched']}")
    print("=" * 72)


if __name__ == "__main__":
    main()

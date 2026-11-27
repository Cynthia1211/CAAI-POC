"""Call the configured external health source and print returned topic chunks."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from external_search import fetch_external_knowledge


QUERY = "What is Type 2 Diabetes "
TOP_K = 2


def main() -> None:
    result = fetch_external_knowledge(QUERY, top_k=TOP_K)

    print("=" * 72)
    print(f"Question: {QUERY}")
    print(f"Searched: {result['searched']}")
    print(f"Triggered: {result['triggered']}")
    if result["status_message"]:
        print(f"Status: {result['status_message']}")
    print(f"Returned results: {len(result['items'])}")

    for index, item in enumerate(result["items"], start=1):
        chunk = item.get("chunk", {})
        print("-" * 72)
        print(f"{index}. Title: {chunk.get('section', '')}")
        print(f"Category: {chunk.get('category', '')}")
        print(f"Source: {chunk.get('source_name', '')}")
        print(f"URL: {chunk.get('url', '')}")
        print(f"Text: {chunk.get('text', '')}")
        print(f"Retrieved via: {item.get('via', [])}")

    print("\nChunks returned for the RAG flow:")
    for chunk in result["chunks"]:
        print(f"Chunk Name: {chunk['Chunk Name']}")
        print(f"Source: {chunk['Source']}")
        print(f"Detail:\n{chunk['Detail']}")
        print(f"Entities: {chunk['Entities']}")
        print("-" * 72)
    print("=" * 72)


if __name__ == "__main__":
    main()

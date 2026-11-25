"""Rebuild doc_chunks from data/data.md without calling build_index.py."""

import sys
from pathlib import Path

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_BACKEND, require_database_url
from embedding_provider import embed_text

DOC_OWNER_USER_ID = "USER-001"


def parse_markdown_chunks(
    md_path: Path,
    user_id: str = DOC_OWNER_USER_ID,
) -> list[dict]:
    """Split Markdown into chunks owned by the specified user."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    category = "General"
    section: str | None = None
    section_lines: list[str] = []
    sections: list[tuple[str, str, list[str]]] = []

    def save_section() -> None:
        if section is not None:
            sections.append((category, section, section_lines.copy()))

    for line in lines:
        if line.startswith("## "):
            save_section()
            category = line[3:].strip()
            section = None
            section_lines = []
        elif line.startswith("### "):
            save_section()
            section = line[4:].strip()
            section_lines = []
        elif section is not None and line.strip() != "---":
            section_lines.append(line)

    save_section()

    chunks = []
    for index, (chunk_category, chunk_section, body_lines) in enumerate(sections, start=1):
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        chunks.append({
            "id": f"chunk_{index:02d}",
            "category": chunk_category,
            "section": chunk_section,
            "text": f"Category: {chunk_category}\nSection: {chunk_section}\n{body}",
            "user_id": user_id,
        })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Generate document embeddings for every parsed chunk."""
    indexed_chunks = []
    for chunk in chunks:
        vector = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")
        indexed_chunks.append({**chunk, "embedding": vector})
        print(
            f"Embedded {chunk['id']} for {chunk['user_id']} [{chunk['category']}] -> "
            f"{chunk['section']} (dim={len(vector)})",
            flush=True,
        )
    return indexed_chunks


def reset_doc_chunks(indexed_chunks: list[dict], database_url: str) -> None:
    """Replace doc_chunks and require an owning user for every indexed chunk."""
    insert_sql = """
        insert into doc_chunks (id, category, section, content, embedding, user_id)
        values (%s, %s, %s, %s, %s::vector, %s)
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE doc_chunks")
            cur.execute("""
                ALTER TABLE doc_chunks
                ADD COLUMN IF NOT EXISTS user_id TEXT
                REFERENCES user_profiles(user_id) ON DELETE CASCADE
            """)
            cur.execute("ALTER TABLE doc_chunks ALTER COLUMN user_id SET NOT NULL")
            for chunk in indexed_chunks:
                vector_literal = "[" + ",".join(
                    str(value) for value in chunk["embedding"]
                ) + "]"
                cur.execute(
                    insert_sql,
                    (
                        chunk["id"],
                        chunk["category"],
                        chunk["section"],
                        chunk["text"],
                        vector_literal,
                        chunk["user_id"],
                    ),
                )


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "data.md"
    database_url = require_database_url()

    chunks = parse_markdown_chunks(input_path, user_id=DOC_OWNER_USER_ID)
    indexed_chunks = embed_chunks(chunks)
    reset_doc_chunks(indexed_chunks, database_url)

    print(
        f"Reset and inserted {len(indexed_chunks)} chunks in "
        f"{DATABASE_BACKEND} table 'doc_chunks'.",
        flush=True,
    )


if __name__ == "__main__":
    main()




"""Rebuild a selected chunk table from a selected Markdown file."""

import sys
from pathlib import Path

import psycopg
from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_BACKEND, require_database_url
from embedding_provider import embed_text

DOC_OWNER_USER_ID: str | None = "USER-001"  # Set to None for unowned/shared chunks.
CHUNK_TABLE = "personal_chunks"  # Choose "doc_chunks" or "personal_chunks".
MARKDOWN_FILE = PROJECT_ROOT / "data" / "personal_data.md"

def parse_markdown_chunks(
    md_path: Path,
    user_id: str | None = DOC_OWNER_USER_ID,
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


def reset_chunk_table(
    indexed_chunks: list[dict],
    database_url: str,
    table_name: str = CHUNK_TABLE,
) -> None:
    """Replace the selected chunk table; require an owning user per chunk."""
    if not indexed_chunks:
        raise ValueError("No chunks were provided; cannot infer the embedding dimension.")

    embedding_dimension = len(indexed_chunks[0]["embedding"])
    if any(len(chunk["embedding"]) != embedding_dimension for chunk in indexed_chunks):
        raise ValueError("All chunks must have embeddings with the same dimension.")

    table = sql.Identifier(table_name)
    create_table_sql = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            section TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector({}) NOT NULL,
            user_id TEXT REFERENCES user_profiles(user_id) ON DELETE CASCADE
        )
    """).format(table, sql.Literal(embedding_dimension))
    insert_sql = sql.SQL("""
        INSERT INTO {} (id, category, section, content, embedding, user_id)
        VALUES (%s, %s, %s, %s, %s::vector, %s)
    """).format(table)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(sql.SQL("TRUNCATE TABLE {}").format(table))
            cur.execute(sql.SQL("""
                ALTER TABLE {}
                ADD COLUMN IF NOT EXISTS user_id TEXT
                REFERENCES user_profiles(user_id) ON DELETE CASCADE
            """).format(table))
            cur.execute(sql.SQL("ALTER TABLE {} ALTER COLUMN user_id DROP NOT NULL").format(table))
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
    database_url = require_database_url()

    chunks = parse_markdown_chunks(MARKDOWN_FILE, user_id=DOC_OWNER_USER_ID)
    indexed_chunks = embed_chunks(chunks)
    reset_chunk_table(indexed_chunks, database_url, table_name=CHUNK_TABLE)

    print(
        f"Reset and inserted {len(indexed_chunks)} chunks from '{MARKDOWN_FILE}' "
        f"into {DATABASE_BACKEND} table '{CHUNK_TABLE}'.",
        flush=True,
    )


if __name__ == "__main__":
    main()




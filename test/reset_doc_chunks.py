"""Rebuild a selected chunk table from a selected Markdown file."""

import sys
import json
import os
from pathlib import Path

import boto3
import psycopg
from psycopg import sql
from botocore import UNSIGNED
from botocore.config import Config

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
    owner_user_id: str | None = DOC_OWNER_USER_ID,
) -> list[dict]:
    """Split Markdown into chunks, retaining an optional graph owner in memory."""
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
            "owner_user_id": owner_user_id,
        })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Generate document embeddings for every parsed chunk."""
    indexed_chunks = []
    for chunk in chunks:
        vector = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")
        indexed_chunks.append({**chunk, "embedding": vector})
        print(
            f"Embedded {chunk['id']} for {chunk['owner_user_id']} [{chunk['category']}] -> "
            f"{chunk['section']} (dim={len(vector)})",
            flush=True,
        )
    return indexed_chunks


def reset_chunk_table(
    indexed_chunks: list[dict],
    database_url: str,
    table_name: str = CHUNK_TABLE,
) -> None:
    """Replace a chunk table with no user_id column and rebuild its graph ownership."""
    if not indexed_chunks:
        raise ValueError("No chunks were provided; cannot infer the embedding dimension.")
    if table_name not in {"doc_chunks", "personal_chunks"}:
        raise ValueError("table_name must be 'doc_chunks' or 'personal_chunks'.")

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
            embedding vector({}) NOT NULL
        )
    """).format(table, sql.Literal(embedding_dimension))
    insert_sql = sql.SQL("""
        INSERT INTO {} (id, category, section, content, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
    """).format(table)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(sql.SQL("ALTER TABLE {} DROP COLUMN IF EXISTS user_id CASCADE").format(table))
            cur.execute(sql.SQL("TRUNCATE TABLE {}").format(table))
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
                    ),
                )

    reset_chunk_graph(indexed_chunks, table_name)


def _neptune_client():
    endpoint = os.getenv("NEPTUNE_ENDPOINT", "https://localhost:8182")
    return boto3.client(
        "neptunedata",
        endpoint_url=endpoint,
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
        config=Config(signature_version=UNSIGNED),
        verify=False if endpoint.startswith("https://localhost:") else True,
    )


def reset_chunk_graph(indexed_chunks: list[dict], table_name: str) -> None:
    """Replace graph Chunk nodes and User-owns-Chunk edges for a chunk table."""
    if table_name not in {"doc_chunks", "personal_chunks"}:
        raise ValueError("table_name must be 'doc_chunks' or 'personal_chunks'.")
    client = _neptune_client()
    client.execute_open_cypher_query(
        openCypherQuery="""MATCH (chunk:Chunk {table_name: $table_name})
        DETACH DELETE chunk""",
        parameters=json.dumps({"table_name": table_name}),
    )
    for chunk in indexed_chunks:
        graph_chunk_id = f"{table_name}:{chunk['id']}"
        parameters = {
            "graph_chunk_id": graph_chunk_id,
            "chunk_id": chunk["id"],
            "table_name": table_name,
            "owner_user_id": chunk.get("owner_user_id"),
        }
        if parameters["owner_user_id"]:
            query = """MERGE (u:User {id: $owner_user_id})
            MERGE (chunk:Chunk {id: $graph_chunk_id})
            SET chunk.chunk_id = $chunk_id, chunk.table_name = $table_name
            MERGE (u)-[:relationship {relationship: 'owns'}]->(chunk)"""
        else:
            query = """MERGE (chunk:Chunk {id: $graph_chunk_id})
            SET chunk.chunk_id = $chunk_id, chunk.table_name = $table_name"""
        client.execute_open_cypher_query(
            openCypherQuery=query,
            parameters=json.dumps(parameters),
        )


def main() -> None:
    database_url = require_database_url()

    chunks = parse_markdown_chunks(MARKDOWN_FILE, owner_user_id=DOC_OWNER_USER_ID)
    indexed_chunks = embed_chunks(chunks)
    reset_chunk_table(indexed_chunks, database_url, table_name=CHUNK_TABLE)

    print(
        f"Reset and inserted {len(indexed_chunks)} chunks from '{MARKDOWN_FILE}' "
        f"into {DATABASE_BACKEND} table '{CHUNK_TABLE}'.",
        flush=True,
    )


if __name__ == "__main__":
    main()



# command to verify the inserted data:
# doc_chunks: python -c 'import psycopg; from config import require_database_url; conn=psycopg.connect(require_database_url()); rows=conn.execute("SELECT id, category, section, content FROM doc_chunks ORDER BY id").fetchall(); print(*rows, sep="\n"); conn.close()'
# personal_chunks: python -c 'import psycopg; from config import require_database_url; conn=psycopg.connect(require_database_url()); rows=conn.execute("SELECT id, category, section, content FROM personal_chunks ORDER BY id").fetchall(); print(*rows, sep="\n"); conn.close()'
# python -c 'import boto3; from botocore import UNSIGNED; from botocore.config import Config; c=boto3.client("neptunedata", endpoint_url="https://localhost:8182", region_name="us-east-2", config=Config(signature_version=UNSIGNED), verify=False); q="MATCH (u:User)-[r:relationship {relationship: '\''owns'\''}]->(chunk:Chunk {table_name: '\''personal_chunks'\''}) RETURN u.id, r.relationship, chunk.chunk_id, chunk.id ORDER BY chunk.chunk_id"; rows=c.execute_open_cypher_query(openCypherQuery=q)["results"]; print(*rows, sep="\n")'

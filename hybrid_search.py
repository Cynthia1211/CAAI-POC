"""Hybrid vector and keyword search over a selected PostgreSQL chunk table."""

import psycopg
from psycopg import sql as psql

from config import require_database_url
from embedding_provider import embed_text as embed_configured_text


def fetch_vector_candidates(
    q_vec: list[float],
    database_url: str,
    table_name: str,
    user_id: str | None,
    candidate_limit: int = 10,
) -> list[dict]:
    """Retrieve nearest chunks from PostgreSQL using pgvector cosine distance."""
    q_literal = "[" + ",".join(str(value) for value in q_vec) + "]"
    query_sql = psql.SQL("""
    SELECT
        id,
        category,
        section,
        content,
        1 - (embedding <=> %s::vector) AS vec_score
    FROM {}
    WHERE user_id IS NOT DISTINCT FROM %s
    ORDER BY embedding <=> %s::vector ASC
    LIMIT %s
    """).format(psql.Identifier(table_name))
    candidates = []
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, (q_literal, user_id, q_literal, candidate_limit))
            for row_id, category, section, content, vec_score in cur.fetchall():
                candidates.append({
                    "id": row_id,
                    "category": category,
                    "section": section,
                    "text": content,
                    "vec_score": float(vec_score) if vec_score is not None else 0.0,
                })
    return candidates


def fetch_keyword_candidates(
    query: str,
    database_url: str,
    table_name: str,
    user_id: str | None,
    candidate_limit: int = 10,
) -> list[dict]:
    """Retrieve chunks matching query terms using PostgreSQL full-text search."""
    query_sql = psql.SQL("""
    WITH query_parts AS (
        SELECT
            regexp_replace(
                websearch_to_tsquery('english', %s)::text,
                ' & ',
                ' | ',
                'g'
            )::tsquery AS tsquery,
            %s::text AS original_query
    ), ranked_chunks AS (
        SELECT
            id,
            category,
            section,
            content,
            document_vector,
            ts_rank_cd(document_vector, query_parts.tsquery) AS raw_kw_score,
            ARRAY(
                SELECT DISTINCT query_term.term
                FROM regexp_split_to_table(
                    query_parts.original_query,
                    '[^[:alnum:]]+'
                ) AS query_term(term)
                WHERE query_term.term <> ''
                  AND document_vector @@ plainto_tsquery('english', query_term.term)
                ORDER BY query_term.term
            ) AS matched_terms
        FROM (
            SELECT
                id,
                category,
                section,
                content,
                user_id,
                to_tsvector('english', coalesce(content, '')) AS document_vector
            FROM {}
        ) AS chunks
        CROSS JOIN query_parts
        WHERE chunks.user_id IS NOT DISTINCT FROM %s
          AND document_vector @@ query_parts.tsquery
    )
    SELECT
        id,
        category,
        section,
        content,
        matched_terms,
        coalesce(raw_kw_score / nullif(max(raw_kw_score) OVER (), 0), 0) AS kw_score
    FROM ranked_chunks
    ORDER BY raw_kw_score DESC
    LIMIT %s
    """).format(psql.Identifier(table_name))
    candidates = []
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, (query, query, user_id, candidate_limit))
            for row_id, category, section, content, matched_terms, kw_score in cur.fetchall():
                candidates.append({
                    "id": row_id,
                    "category": category,
                    "section": section,
                    "text": content,
                    "kw_score": float(kw_score) if kw_score is not None else 0.0,
                    "matched_kws": matched_terms or [],
                })
    return candidates


def hybrid_search(
    query: str,
    table_name: str,
    user_id: str | None,
    top_k: int = 3,
    vec_weight: float = 0.8,
    kw_weight: float = 0.2,
    candidate_limit: int = 10,
) -> list[dict]:
    """Search PostgreSQL by vector and keywords, then return ranked chunks."""
    database_url = require_database_url()
    q_vec = embed_configured_text(query, "RETRIEVAL_QUERY")
    limit = max(candidate_limit, top_k * 3)

    vector_candidates = fetch_vector_candidates(
        q_vec,
        database_url,
        table_name=table_name,
        user_id=user_id,
        candidate_limit=limit,
    )
    keyword_candidates = fetch_keyword_candidates(
        query,
        database_url,
        table_name=table_name,
        user_id=user_id,
        candidate_limit=limit,
    )

    if not vector_candidates and not keyword_candidates:
        return []

    merged: dict[str, dict] = {}
    for item in vector_candidates:
        merged[item["id"]] = {
            "id": item["id"],
            "category": item["category"],
            "section": item["section"],
            "text": item["text"],
            "vec_score": item["vec_score"],
            "kw_score": 0.0,
            "matched_kws": [],
        }
    for item in keyword_candidates:
        merged.setdefault(item["id"], {
            "id": item["id"],
            "category": item["category"],
            "section": item["section"],
            "text": item["text"],
            "vec_score": 0.0,
            "kw_score": 0.0,
            "matched_kws": [],
        })
        merged[item["id"]]["kw_score"] = item["kw_score"]
        merged[item["id"]]["matched_kws"] = item["matched_kws"]

    ranked_chunks = []
    for item in merged.values():
        score = vec_weight * item["vec_score"] + kw_weight * item["kw_score"]
        ranked_chunks.append({**item, "score": score})

    ranked_chunks.sort(key=lambda chunk: chunk["score"], reverse=True)
    chunks = []
    for item in ranked_chunks[:top_k]:
        chunks.append({
            "Chunk Name": item["id"],
            "Source": (
                f"[DIRECT | vec={item['vec_score']:.4f} | "
                f"kw={item['kw_score']:.4f} | rank_score={item['score']:.4f}]"
            ),
            "Category": item["category"],
            "Section": item["section"],
            "Matched": ", ".join(item["matched_kws"]) or "None",
            "Details": item["text"],
        })
    return chunks

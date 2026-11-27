"""Hybrid vector and keyword search over PostgreSQL chunk tables."""

import psycopg
from psycopg import sql as psql

from chunk_builder import build_chunks
from config import require_database_url
from embedding_provider import embed_text as embed_configured_text


def _fetch_all(
    query: psql.Composed,
    params: tuple,
    database_url: str,
    conn: psycopg.Connection | None,
) -> list[tuple]:
    if conn is not None:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()


def fetch_vector_candidates(
    q_vec: list[float],
    database_url: str,
    table_name: str,
    candidate_limit: int = 10,
    candidate_ids: list[str] | None = None,
    conn: psycopg.Connection | None = None,
) -> list[dict]:
    """Retrieve nearest chunks, optionally restricted to candidate IDs."""
    if candidate_ids is not None and not candidate_ids:
        return []

    q_literal = "[" + ",".join(str(value) for value in q_vec) + "]"
    where_clause = (
        psql.SQL("WHERE id = ANY(%s::text[])")
        if candidate_ids is not None
        else psql.SQL("")
    )
    query = psql.SQL("""
        SELECT id, category, section, content,
               1 - (embedding <=> %s::vector) AS vec_score
        FROM {}
        {}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s
    """).format(psql.Identifier(table_name), where_clause)
    params = (
        (q_literal, candidate_ids, q_literal, candidate_limit)
        if candidate_ids is not None
        else (q_literal, q_literal, candidate_limit)
    )
    rows = _fetch_all(query, params, database_url, conn)
    return [
        {
            "id": row_id,
            "category": category,
            "section": section,
            "text": content,
            "vec_score": float(vec_score or 0.0),
        }
        for row_id, category, section, content, vec_score in rows
    ]


def fetch_keyword_candidates(
    query: str,
    database_url: str,
    table_name: str,
    candidate_limit: int = 10,
    candidate_ids: list[str] | None = None,
    conn: psycopg.Connection | None = None,
) -> list[dict]:
    """Retrieve full-text matches, optionally restricted to candidate IDs."""
    if candidate_ids is not None and not candidate_ids:
        return []

    where_clause = (
        psql.SQL("WHERE id = ANY(%s::text[])")
        if candidate_ids is not None
        else psql.SQL("")
    )
    statement = psql.SQL("""
        WITH query_parts AS (
            SELECT
                regexp_replace(
                    websearch_to_tsquery('english', %s)::text,
                    ' & ', ' | ', 'g'
                )::tsquery AS tsquery,
                %s::text AS original_query
        ), ranked_chunks AS (
            SELECT id, category, section, content, document_vector,
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
                SELECT id, category, section, content,
                       to_tsvector('english', coalesce(content, '')) AS document_vector
                FROM {}
                {}
            ) AS chunks
            CROSS JOIN query_parts
            WHERE document_vector @@ query_parts.tsquery
        )
        SELECT id, category, section, content, matched_terms,
               coalesce(raw_kw_score / nullif(max(raw_kw_score) OVER (), 0), 0) AS kw_score
        FROM ranked_chunks
        ORDER BY raw_kw_score DESC
        LIMIT %s
    """).format(psql.Identifier(table_name), where_clause)
    params = (
        (query, query, candidate_ids, candidate_limit)
        if candidate_ids is not None
        else (query, query, candidate_limit)
    )
    rows = _fetch_all(statement, params, database_url, conn)
    return [
        {
            "id": row_id,
            "category": category,
            "section": section,
            "text": content,
            "kw_score": float(kw_score or 0.0),
            "matched_kws": list(matched_terms or []),
        }
        for row_id, category, section, content, matched_terms, kw_score in rows
    ]


def _merge_and_rank_candidates(
    vector_candidates: list[dict],
    keyword_candidates: list[dict],
    vec_weight: float,
    kw_weight: float,
) -> list[dict]:
    """Merge vector and keyword results by ID and sort by combined score."""
    merged: dict[str, dict] = {}
    for candidate in vector_candidates:
        merged[candidate["id"]] = {
            **candidate,
            "vec_score": float(candidate.get("vec_score", 0.0)),
            "kw_score": 0.0,
            "matched_kws": [],
        }

    for candidate in keyword_candidates:
        item = merged.setdefault(candidate["id"], {
            **candidate,
            "vec_score": 0.0,
            "kw_score": 0.0,
            "matched_kws": [],
        })
        item.update({
            "category": candidate.get("category", item.get("category")),
            "section": candidate.get("section", item.get("section")),
            "text": candidate.get("text", item.get("text", "")),
            "kw_score": float(candidate.get("kw_score", 0.0)),
            "matched_kws": list(candidate.get("matched_kws", [])),
        })

    ranked = []
    for candidate in merged.values():
        candidate["score"] = (
            vec_weight * candidate.get("vec_score", 0.0)
            + kw_weight * candidate.get("kw_score", 0.0)
        )
        ranked.append(candidate)
    return sorted(ranked, key=lambda candidate: candidate["score"], reverse=True)


def _build_hybrid_chunks(
    candidates: list[dict],
    table_name: str,
    top_k: int,
) -> list[dict]:
    query_results = []
    selected = candidates[:top_k]
    for candidate in selected:
        vec_score = float(candidate.get("vec_score", 0.0))
        kw_score = float(candidate.get("kw_score", 0.0))
        score = float(candidate.get("score", 0.0))
        matched = ", ".join(candidate.get("matched_kws", [])) or "None"
        query_results.append({
            "source": (
                f"{table_name} ({candidate['id']}) "
                f"[vec={vec_score:.4f} | kw={kw_score:.4f} | "
                f"rank_score={score:.4f} | matched: {matched}]"
            ),
            "records": [{"fields": {
                "category": candidate.get("category", ""),
                "section": candidate.get("section", ""),
                "content": candidate.get("text", ""),
            }}],
        })

    return build_chunks(
        query_results,
        chunk_prefix="Hybrid",
        empty_detail="No matching chunks returned.",
    )


def hybrid_search(
    query: str,
    table_name: str,
    top_k: int = 3,
    vec_weight: float = 0.8,
    kw_weight: float = 0.2,
    candidate_limit: int = 10,
    candidate_ids: list[str] | None = None,
) -> list[dict]:
    """Run vector and keyword retrieval, merge scores, and build ranked chunks."""
    if candidate_ids is not None and not candidate_ids:
        return []

    database_url = require_database_url()
    query_vector = embed_configured_text(query, "RETRIEVAL_QUERY")
    limit = max(candidate_limit, top_k * 3)
    with psycopg.connect(database_url) as conn:
        vector_candidates = fetch_vector_candidates(
            query_vector,
            database_url,
            table_name,
            candidate_limit=limit,
            candidate_ids=candidate_ids,
            conn=conn,
        )
        keyword_candidates = fetch_keyword_candidates(
            query,
            database_url,
            table_name,
            candidate_limit=limit,
            candidate_ids=candidate_ids,
            conn=conn,
        )

    if not vector_candidates and not keyword_candidates:
        return []
    ranked_candidates = _merge_and_rank_candidates(
        vector_candidates,
        keyword_candidates,
        vec_weight=vec_weight,
        kw_weight=kw_weight,
    )
    return _build_hybrid_chunks(ranked_candidates, table_name, top_k)

"""LLM extraction and safe, parameterized queries for the structured data tables."""

from datetime import date
from typing import Literal

import psycopg
from pydantic import BaseModel, Field, ValidationError

from chunk_builder import build_chunks
from config import (
    BEDROCK_EXTRACTOR_MODEL_ID,
    BEDROCK_REGION,
    require_database_url,
)
from llm_provider import generate_text, parse_json_object


TableName = Literal["user_profiles", "family_relationships", "appointments"]
current_user = "USER-001"
DEFAULT_CURRENT_USER = current_user


class StructuredQuerySpec(BaseModel):
    """Validated search plan extracted from a natural-language question."""

    tables: list[TableName] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None


def extract_structured_query(
    query: str,
    reference_date: str | None = None,
) -> StructuredQuerySpec:
    today = reference_date or date.today().isoformat()
    prompt = f"""
You extract a safe search plan for read-only PostgreSQL lookups.
Today is {today}.

PostgreSQL tables and columns:
- user_profiles: user_id, user_name
- family_relationships: family_id, family_name, family_relationship, user_id
- appointments: appointment_id, appointment_desc, appointment_date, user_id

Rules:
- Select only PostgreSQL tables relevant to the question. If unsure, select all three tables.
- search_terms should contain names, IDs, and meaningful phrases explicitly mentioned in the question. Do not include generic words such as "appointment" or "family" by themselves.
- Resolve relative appointment dates using today's date. Use YYYY-MM-DD for date_from/date_to. Leave dates null if no date filter is requested.
- Do not write SQL. Return only the structured filters requested by the schema.

Question: {query}
""".strip()

    instruction = """
Return exactly one JSON object with this shape:
{"tables": ["appointments"], "search_terms": [], "date_from": null, "date_to": null}
Only use allowed table names. Use an empty tables array to mean all available tables.
""".strip()
    last_error: ValueError | None = None
    for attempt in range(2):
        retry = (
            "\nReturn valid JSON only. Quote strings with double quotes and use null for missing dates."
            if attempt
            else ""
        )
        output = generate_text(
            model_id=BEDROCK_EXTRACTOR_MODEL_ID,
            region_name=BEDROCK_REGION,
            prompt=f"{prompt}\n\n{instruction}{retry}",
        )
        try:
            return StructuredQuerySpec.model_validate(parse_json_object(output))
        except (ValueError, ValidationError) as exc:
            last_error = exc
    raise ValueError("The query extractor did not return a valid search plan.") from last_error


TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "user_profiles": ("user_id", "user_name"),
    "family_relationships": ("family_id", "family_name", "family_relationship"),
    "appointments": ("appointment_id", "appointment_desc", "appointment_date"),
}

SEARCH_COLUMNS: dict[str, tuple[str, ...]] = {
    "user_profiles": ("user_id", "user_name"),
    "family_relationships": ("family_id", "family_name", "family_relationship"),
    "appointments": ("appointment_id", "appointment_desc"),
}

DATE_COLUMNS: dict[str, str] = {
    "appointments": "appointment_date",
}


def build_postgres_queries(
    spec: StructuredQuerySpec,
    current_user: str = DEFAULT_CURRENT_USER,
) -> list[dict[str, object]]:
    """Build the allowlisted parameterized SQL statements selected by the extractor."""
    selected_tables = list(dict.fromkeys(spec.tables or list(TABLE_COLUMNS)))
    sql_statements: list[dict[str, object]] = []
    for table in selected_tables:
        columns = TABLE_COLUMNS[table]
        clauses: list[str] = ["user_id = %s"]
        params: list[object] = [current_user]

        if spec.search_terms:
            search_clauses = []
            for term in spec.search_terms:
                for column in SEARCH_COLUMNS[table]:
                    search_clauses.append(f"{column}::text ILIKE %s")
                    params.append(f"%{term.strip()}%")
            clauses.append("(" + " OR ".join(search_clauses) + ")")

        date_column = DATE_COLUMNS.get(table)
        if date_column:
            if spec.date_from:
                clauses.append(f"{date_column}::date >= %s::date")
                params.append(spec.date_from)
            if spec.date_to:
                clauses.append(f"{date_column}::date <= %s::date")
                params.append(spec.date_to)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT {', '.join(columns)} FROM {table}{where} "
            f"ORDER BY {columns[0]} LIMIT 20"
        )
        sql_statements.append({
            "table": table,
            "columns": columns,
            "query": sql,
            "params": params,
        })
    return sql_statements


def query_postgres_tables(
    spec: StructuredQuerySpec,
    database_url: str | None = None,
    current_user: str = DEFAULT_CURRENT_USER,
) -> list[dict]:
    """Execute selected SQL queries and return one chunk per query."""
    dsn = database_url or require_database_url()
    query_results: list[dict] = []
    sql_statements = build_postgres_queries(spec, current_user=current_user)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for statement in sql_statements:
                table = statement["table"]
                columns = statement["columns"]
                cur.execute(statement["query"], statement["params"])
                query_records: list[dict] = []
                for row in cur.fetchall():
                    record = {
                        "source": table,
                        "fields": dict(zip(columns, row)),
                    }
                    query_records.append(record)
                query_results.append({"source": table, "records": query_records})

    return build_chunks(
        query_results,
        chunk_prefix="PostgreSQL",
        empty_detail="No rows returned.",
        entity_fields={
            "user_profiles": "user_id",
            "family_relationships": "family_id",
            "appointments": "appointment_id",
        },
    )


def run_structured_query(
    query: str,
    reference_date: str | None = None,
    database_url: str | None = None,
    current_user: str = DEFAULT_CURRENT_USER,
) -> list[dict]:
    """Extract SQL filters, execute selected queries, and return only their chunks."""
    spec = extract_structured_query(query=query, reference_date=reference_date)
    return query_postgres_tables(
        spec,
        database_url=database_url,
        current_user=current_user,
    )

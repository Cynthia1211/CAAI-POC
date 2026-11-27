"""Safe PostgreSQL record lookups by primary key IDs."""

import psycopg
from psycopg import sql as psql

from config import require_database_url


def get_records_by_ids(
    table_name: str,
    ids: list[str],
    id_column: str = "id",
    columns: tuple[str, ...] | list[str] | None = None,
    database_url: str | None = None,
) -> dict | None:
    """Return rows for a list of IDs from a table with SQL-safe identifiers."""
    if not ids:
        return None
    if columns is not None and not columns:
        raise ValueError("columns must contain at least one column or be None.")

    selected_ids = sorted(set(str(item) for item in ids))
    selected_columns = (
        psql.SQL(", ").join(psql.Identifier(column) for column in columns)
        if columns is not None
        else psql.SQL("*")
    )
    query = psql.SQL("""
        SELECT {} FROM {}
        WHERE {} = ANY(%s::text[])
    """).format(
        selected_columns,
        psql.Identifier(table_name),
        psql.Identifier(id_column),
    )

    dsn = database_url or require_database_url()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (selected_ids,))
            rows = cursor.fetchall()
            if not rows:
                return None
            column_names = [description.name for description in cursor.description]

    return {
        "source": table_name,
        "records": [
            {"fields": dict(zip(column_names, row))}
            for row in rows
        ],
    }

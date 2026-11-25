"""Convert SQL and graph search results to the shared chunk structure."""

from datetime import date, datetime
from typing import Any


def _value_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return "N/A" if value is None else str(value)


def build_chunks(
    query_results: list[dict],
    chunk_prefix: str,
    empty_detail: str,
    entity_fields: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build shared-format chunks from SQL query or graph lookup records."""
    chunks: list[dict[str, Any]] = []
    for index, query_result in enumerate(query_results, start=1):
        source = query_result["source"]
        records = query_result["records"]
        entity_field = (entity_fields or {}).get(source)
        entities: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        row_details: list[str] = []
        for row_index, record in enumerate(records, start=1):
            fields = record["fields"]
            row_details.append(
                f"{row_index}. " + "; ".join(
                    f"{name}: {_value_text(value)}" for name, value in fields.items()
                )
            )
            entity_value = fields.get(entity_field) if entity_field else None
            if entity_value is not None and str(entity_value) not in seen_ids:
                seen_ids.add(str(entity_value))
                entities.append({entity_field: str(entity_value)})
        relation_details = query_result.get("graph_relations", [])
        if relation_details:
            row_details.append("Graph relations: " + "; ".join(relation_details))
        detail = "\n".join(row_details) if row_details else empty_detail
        chunks.append({
            "Chunk Name": f"{chunk_prefix}{index:02d}",
            "Source": source,
            "Detail": detail,
            "Entities": entities,
        })
    return chunks

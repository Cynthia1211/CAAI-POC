import json
import os
import boto3
import psycopg
from botocore import UNSIGNED
from botocore.config import Config

from chunk_builder import build_chunks
from config import require_database_url


# Entity field -> graph label and allowlisted PostgreSQL lookup definition.
ENTITY_SQL_MAP = {
    "appointment_id": {
        "graph_label": "Appointment",
        "table": "appointments",
        "primary_key": "appointment_id",
        "columns": ("appointment_id", "appointment_desc", "appointment_date"),
    },
    "user_id": {
        "graph_label": "User",
        "table": "user_profiles",
        "primary_key": "user_id",
        "columns": ("user_id", "user_name"),
    },
    "family_id": {
        "graph_label": "Family",
        "table": "family_relationships",
        "primary_key": "family_id",
        "columns": ("family_id", "family_name", "family_relationship"),
    },
}


def _neptune_client():
    endpoint = os.getenv("NEPTUNE_ENDPOINT", "https://localhost:8182")
    return boto3.client(
        "neptunedata",
        endpoint_url=endpoint,
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
        config=Config(signature_version=UNSIGNED),
        verify=False if endpoint.startswith("https://localhost:") else True,
    )


def extract_entities_from_chunks(chunks: list[dict]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        for entity in chunk.get("Entities", []):
            for entity_type, entity_id in entity.items():
                if entity_type not in ENTITY_SQL_MAP or entity_id is None:
                    continue
                key = (entity_type, str(entity_id))
                if key not in seen:
                    seen.add(key)
                    entities.append({entity_type: str(entity_id)})
    return entities


def find_one_hop_entities(seed_entities: list[dict[str, str]]) -> dict:
    """Find outgoing one-hop Neptune neighbors from typed seed entities."""
    if not seed_entities:
        return {"seed_entities": [], "entities": [], "relations": []}

    ids_by_type: dict[str, list[str]] = {}
    for entity in seed_entities:
        entity_type, entity_id = next(iter(entity.items()))
        ids_by_type.setdefault(entity_type, []).append(entity_id)

    client = _neptune_client()
    relations: list[dict] = []
    related_entities: list[dict[str, str]] = []
    seen_relations: set[tuple] = set()
    seen_related: set[tuple[str, str]] = set()
    label_to_entity_type = {
        details["graph_label"]: entity_type
        for entity_type, details in ENTITY_SQL_MAP.items()
    }

    for entity_type, entity_ids in ids_by_type.items():
        graph_label = ENTITY_SQL_MAP[entity_type]["graph_label"]
        query = f"""MATCH (seed:{graph_label})-[r]->(neighbor)
        WHERE seed.id IN $ids AND type(r) IN ['attendedBy', 'relationship']
        RETURN seed.id AS seed_id, labels(seed) AS seed_labels,
               type(r) AS relation_type, properties(r) AS relation_properties,
               neighbor.id AS neighbor_id, labels(neighbor) AS neighbor_labels"""
        response = client.execute_open_cypher_query(
            openCypherQuery=query,
            parameters=json.dumps({"ids": entity_ids}),
        )
        for row in response.get("results", []):
            seed_id = str(row.get("seed_id"))
            neighbor_id = str(row.get("neighbor_id"))
            seed_labels = row.get("seed_labels") or [graph_label]
            neighbor_labels = row.get("neighbor_labels") or []
            relation_type = row.get("relation_type")
            properties = row.get("relation_properties") or {}
            neighbor_type = next(
                (label_to_entity_type[label] for label in neighbor_labels if label in label_to_entity_type),
                None,
            )
            if not neighbor_type:
                continue

            # Normalize edge direction and endpoints to the stored graph schema.
            if relation_type == "relationship":
                if "User" in seed_labels:
                    user_id, family_id = seed_id, neighbor_id
                else:
                    user_id, family_id = neighbor_id, seed_id
                value = properties.get("relationship")
                source_entity = {"user_id": user_id}
                target_entity = {"family_id": family_id}
            elif relation_type == "attendedBy":
                if "Appointment" in seed_labels:
                    appointment_id = seed_id
                    attendee_id = neighbor_id
                    attendee_type = neighbor_type
                else:
                    appointment_id = neighbor_id
                    attendee_id = seed_id
                    attendee_type = entity_type
                value = properties.get("attendedBy")
                source_entity = {"appointment_id": appointment_id}
                target_field = "user_id" if attendee_type == "user_id" else "family_id"
                target_entity = {target_field: attendee_id}
            else:
                continue

            relation_key = (
                relation_type,
                tuple(source_entity.items()),
                tuple(target_entity.items()),
                value,
            )
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                relations.append({
                    "source": relation_type,
                    "value": value,
                    "from": source_entity,
                    "to": target_entity,
                })

            neighbor_key = (neighbor_type, neighbor_id)
            if neighbor_key not in seen_related:
                seen_related.add(neighbor_key)
                related_entities.append({neighbor_type: neighbor_id})

    return {
        "seed_entities": seed_entities,
        "entities": related_entities,
        "relations": relations,
    }


def group_entities_by_table(entities: list[dict[str, str]]) -> list[dict]:
    """Deduplicate typed entities and group them by their mapped SQL table."""
    groups: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        for entity_type, entity_id in entity.items():
            mapping = ENTITY_SQL_MAP.get(entity_type)
            if not mapping or not entity_id:
                continue
            key = (entity_type, str(entity_id))
            if key in seen:
                continue
            seen.add(key)
            table = mapping["table"]
            group = groups.setdefault(table, {"table": table, "entities": []})
            group["entities"].append({entity_type: str(entity_id)})
    return list(groups.values())


def query_records_for_entity_groups(
    groups: list[dict],
    relations: list[dict] | None = None,
) -> list[dict]:
    """Look up one-hop entity records in PostgreSQL, grouped by mapped table."""
    if not groups:
        return []

    query_results: list[dict] = []
    with psycopg.connect(require_database_url()) as conn:
        with conn.cursor() as cur:
            for group in groups:
                table_entities = group["entities"]
                entity_type = next(iter(table_entities[0]))
                entity_ids = {next(iter(entity.values())) for entity in table_entities}
                mapping = ENTITY_SQL_MAP[entity_type]
                columns = mapping["columns"]
                placeholders = ", ".join(["%s"] * len(entity_ids))
                sql = (
                    f"SELECT {', '.join(columns)} FROM {mapping['table']} "
                    f"WHERE {mapping['primary_key']} IN ({placeholders}) "
                    f"ORDER BY {mapping['primary_key']}"
                )
                params = sorted(entity_ids)
                cur.execute(sql, params)
                records = [
                    {"fields": dict(zip(columns, row))}
                    for row in cur.fetchall()
                ]
                relation_details = []
                for relation in relations or []:
                    endpoints = (relation.get("from", {}), relation.get("to", {}))
                    if any(
                        endpoint.get(entity_type) in entity_ids
                        for endpoint in endpoints
                    ):
                        source_id = next(iter(relation.get("from", {}).values()), "")
                        target_id = next(iter(relation.get("to", {}).values()), "")
                        relation_details.append(
                            f"{source_id} -[{relation['source']}: {relation.get('value')}]-> {target_id}"
                        )
                query_results.append({
                    "source": mapping["table"],
                    "records": records,
                    "graph_relations": relation_details,
                })
    return query_results


def search_neptune_relationships(
    sql_chunks: list[dict],
) -> list[dict]:
    """Search one graph hop, load related PostgreSQL rows, and return only chunks."""
    seed_entities = extract_entities_from_chunks(sql_chunks)
    graph_result = find_one_hop_entities(seed_entities)
    grouped_entities = group_entities_by_table(graph_result["entities"])
    record_groups = query_records_for_entity_groups(
        grouped_entities,
        relations=graph_result["relations"],
    )
    record_groups = [group for group in record_groups if group["records"]]
    if not record_groups:
        return []

    return build_chunks(
        record_groups,
        chunk_prefix="Graph",
        empty_detail="No related rows returned.",
    )

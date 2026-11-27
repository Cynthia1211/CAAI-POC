"""Retrieve user-related Neptune entities, then hydrate their PostgreSQL rows."""

import json
import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from chunk_builder import build_chunks
from postgreSQL_search import get_records_by_ids


def _neptune_client():
    endpoint = os.getenv("NEPTUNE_ENDPOINT", "https://localhost:8182")
    return boto3.client(
        "neptunedata",
        endpoint_url=endpoint,
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
        config=Config(signature_version=UNSIGNED),
        verify=False if endpoint.startswith("https://localhost:") else True,
    )


def find_user_related_entities(user_id: str) -> dict[str, list[str]]:
    """Find family, appointment, and owned personal chunk IDs within three hops."""
    client = _neptune_client()
    response = client.execute_open_cypher_query(
        openCypherQuery="""MATCH p=(u:User {id: $user_id})-[:relationship*1..3]-(n)
        WHERE n:Family OR n:Appointment
        RETURN DISTINCT labels(n) AS labels, n.id AS entity_id,
               [edge IN relationships(p) | {
                   from_id: startNode(edge).id,
                   to_id: endNode(edge).id,
                   relationship: edge.relationship
               }] AS path_relations""",
        parameters=json.dumps({"user_id": user_id}),
    )
    entity_ids: dict[str, set[str]] = {
        "family_relationships": set(),
        "appointments": set(),
    }
    graph_relations: set[str] = set()
    for row in response.get("results", []):
        for edge in row.get("path_relations") or []:
            source_id = edge.get("from_id")
            target_id = edge.get("to_id")
            relationship = edge.get("relationship")
            if source_id is not None and target_id is not None:
                graph_relations.add(
                    f"{source_id} -[{relationship or 'relationship'}]-> {target_id}"
                )
        entity_id = row.get("entity_id")
        if entity_id is None:
            continue
        labels = row.get("labels") or []
        if "Family" in labels:
            entity_ids["family_relationships"].add(str(entity_id))
        elif "Appointment" in labels:
            entity_ids["appointments"].add(str(entity_id))

    personal_response = client.execute_open_cypher_query(
        openCypherQuery="""MATCH (u:User {id: $user_id})-[:relationship {relationship: 'owns'}]->
        (chunk:Chunk {table_name: 'personal_chunks'})
        RETURN DISTINCT chunk.chunk_id AS chunk_id, u.id AS user_id""",
        parameters=json.dumps({"user_id": user_id}),
    )
    entity_ids["personal_chunks"] = set()
    for row in personal_response.get("results", []):
        chunk_id = row.get("chunk_id")
        if chunk_id is not None:
            entity_ids["personal_chunks"].add(str(chunk_id))
            graph_relations.add(f"{row.get('user_id')} -[owns]-> {chunk_id}")
    return {
        **{table: sorted(ids) for table, ids in entity_ids.items()},
        "graph_relations": sorted(graph_relations),
    }


def _group_graph_records(ids_by_table: dict[str, list[str]]) -> list[dict]:
    result = []
    for table_name, id_column in (
        ("family_relationships", "family_id"),
        ("appointments", "appointment_id"),
    ):
        group = get_records_by_ids(
            table_name,
            ids_by_table.get(table_name, []),
            id_column=id_column,
        )
        if group:
            entity_ids = ids_by_table.get(table_name, [])
            group["graph_relations"] = [
                relation
                for relation in ids_by_table.get("graph_relations", [])
                if any(entity_id in relation for entity_id in entity_ids)
            ]
            result.append(group)
    return result


def search_user_graph(user_id: str) -> dict:
    """Load user graph neighbors and PostgreSQL rows, returning chunks and candidates."""
    ids_by_table = find_user_related_entities(user_id)
    record_groups = _group_graph_records(ids_by_table)
    graph_chunks = build_chunks(
        record_groups,
        chunk_prefix="Graph",
        empty_detail="No related rows returned.",
        entity_fields={
            "family_relationships": "family_id",
            "appointments": "appointment_id",
        },
    )
    return {
        "chunks": graph_chunks,
        "personal_chunk_ids": ids_by_table["personal_chunks"],
    }


def extract_entities_from_chunks(chunks: list[dict]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        for entity in chunk.get("Entities", []):
            for entity_type, entity_id in entity.items():
                key = (entity_type, str(entity_id))
                if entity_id is not None and key not in seen:
                    seen.add(key)
                    entities.append({entity_type: str(entity_id)})
    return entities

"""Create family rows and user-to-family edges in Neptune."""

import json
import os
import sys
from pathlib import Path

import boto3
import psycopg
from botocore import UNSIGNED
from botocore.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import require_database_url


# Edit these values to load a different user and family member.
FAMILY_RELATIONSHIPS = [
    ("FAM-001", "Linda Chen", "daughter", "USER-001"),
    ("FAM-002", "Lucia Chen", "daughter", "USER-001"),
    ("FAM-003", "Jason Ma", "son", "USER-002"),
]


def _neptune_client():
    # The SSH tunnel forwards local port 8182 to the Neptune cluster endpoint.
    endpoint = os.getenv("NEPTUNE_ENDPOINT", "https://localhost:8182")
    return boto3.client(
        "neptunedata",
        endpoint_url=endpoint,
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
        config=Config(signature_version=UNSIGNED),
        # The local SSH-forwarded hostname does not match Neptune's TLS certificate.
        verify=False if endpoint.startswith("https://localhost:") else True,
    )


def main() -> None:
    client = _neptune_client()
    with psycopg.connect(require_database_url()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS family_relationships (
                family_id TEXT PRIMARY KEY,
                family_name TEXT NOT NULL,
                family_relationship TEXT NOT NULL
            )
        """)
        conn.execute("ALTER TABLE family_relationships DROP COLUMN IF EXISTS user_id CASCADE")
        conn.cursor().executemany(
            """
            INSERT INTO family_relationships
                (family_id, family_name, family_relationship)
            VALUES (%s, %s, %s)
            ON CONFLICT (family_id) DO UPDATE SET
                family_name = EXCLUDED.family_name,
                family_relationship = EXCLUDED.family_relationship
            """,
            [(family_id, family_name, relationship)
             for family_id, family_name, relationship, _user_id in FAMILY_RELATIONSHIPS],
        )

    for family_id, family_name, relationship, user_id in FAMILY_RELATIONSHIPS:
        parameters = json.dumps({
            "user_id": user_id,
            "family_id": family_id,
            "relationship": relationship,
        })
        client.execute_open_cypher_query(
            openCypherQuery="""MATCH (u:User {id: $user_id})-[r:relationship]->(f:Family {id: $family_id})
            DELETE r""",
            parameters=parameters,
        )
        client.execute_open_cypher_query(
            openCypherQuery="""MERGE (u:User {id: $user_id})
            MERGE (f:Family {id: $family_id})
            REMOVE f.name
            MERGE (u)-[:relationship {relationship: $relationship}]->(f)""",
            parameters=parameters,
        )

    print(f"Inserted/updated {len(FAMILY_RELATIONSHIPS)} family row(s) and Neptune relationship(s).")


if __name__ == "__main__":
    main()

# command to verify the inserted data:
# python -c 'import psycopg; from config import require_database_url; conn=psycopg.connect(require_database_url()); print(*conn.execute("SELECT * FROM family_relationships ORDER BY family_id").fetchall(), sep="\n"); conn.close()'
# python -c 'import boto3; from botocore import UNSIGNED; from botocore.config import Config; c=boto3.client("neptunedata", endpoint_url="https://localhost:8182", region_name="us-east-2", config=Config(signature_version=UNSIGNED), verify=False); rows=c.execute_open_cypher_query(openCypherQuery="MATCH (u:User)-[r:relationship]->(f:Family) RETURN u.id, r.relationship, f.id")["results"]; print(*rows, sep="\n")'

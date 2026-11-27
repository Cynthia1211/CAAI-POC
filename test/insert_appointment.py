"""Create appointment rows and user/family-to-appointment Neptune edges."""

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


# Edit these values to load a different appointment, user, or attending family member.
APPOINTMENTS = [
    ("APT-001", "Routine clinic follow-up", "2026-11-02 09:30", "USER-001", "FAM-001"),
    ("APT-002", "Routine clinic follow-up", "2026-12-02 09:30", "USER-001", "FAM-002"),
    ("APT-003", "Routine clinic follow-up", "2027-01-02 09:30", "USER-001", "FAM-001"),
    ("APT-004", "Family visit", "2026-12-22 09:30", "USER-002", "FAM-003"),
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
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id TEXT PRIMARY KEY,
                appointment_desc TEXT NOT NULL,
                appointment_date TIMESTAMP NOT NULL
            )
        """)
        conn.execute("ALTER TABLE appointments DROP COLUMN IF EXISTS user_id CASCADE")
        # The attending family ID stays in this script's input for the Neptune edge.
        conn.cursor().executemany(
            """
            INSERT INTO appointments
                (appointment_id, appointment_desc, appointment_date)
            VALUES (%s, %s, %s)
            ON CONFLICT (appointment_id) DO UPDATE SET
                appointment_desc = EXCLUDED.appointment_desc,
                appointment_date = EXCLUDED.appointment_date
            """,
            [(appointment_id, description, appointment_date)
             for appointment_id, description, appointment_date, user_id, _family_id in APPOINTMENTS],
        )

    for appointment_id, description, appointment_date, user_id, family_id in APPOINTMENTS:
        parameters = json.dumps({"appointment_id": appointment_id})
        # Remove this appointment's previous graph shape before recreating its two edges.
        client.execute_open_cypher_query(
            openCypherQuery="""MATCH (a:Appointment {id: $appointment_id})-[r]-()
            DELETE r""",
            parameters=parameters,
        )
        client.execute_open_cypher_query(
            openCypherQuery="MATCH (a:Appointment {id: $appointment_id}) DELETE a",
            parameters=parameters,
        )
        client.execute_open_cypher_query(
            openCypherQuery="""MERGE (a:Appointment {id: $appointment_id})
            REMOVE a.description, a.date
            MERGE (u:User {id: $user_id})
            MERGE (f:Family {id: $family_id})
            MERGE (u)-[:relationship {relationship: 'scheduled'}]->(a)
            MERGE (f)-[:relationship {relationship: 'accompanies'}]->(a)""",
            parameters=json.dumps({
                "appointment_id": appointment_id,
                "user_id": user_id,
                "family_id": family_id,
            }),
        )

    print(f"Inserted/updated {len(APPOINTMENTS)} appointment row(s) and Neptune relationship(s).")


if __name__ == "__main__":
    main()

# command to verify the inserted data:
# python -c 'import psycopg; from config import require_database_url; conn=psycopg.connect(require_database_url()); rows=conn.execute("SELECT appointment_id, appointment_desc, appointment_date FROM appointments ORDER BY appointment_id").fetchall(); print(*[(a, b, c.strftime("%Y-%m-%d %H:%M:%S")) for a,b,c in rows], sep="\n"); conn.close()'
# python -c 'import boto3; from botocore import UNSIGNED; from botocore.config import Config; c=boto3.client("neptunedata", endpoint_url="https://localhost:8182", region_name="us-east-2", config=Config(signature_version=UNSIGNED), verify=False); q="MATCH (x)-[r:relationship]->(a:Appointment) RETURN labels(x), x.id, r.relationship, a.id ORDER BY a.id, r.relationship"; rows=c.execute_open_cypher_query(openCypherQuery=q)["results"]; print(*rows, sep="\n")'

"""Create user_profiles if needed and insert an editable sample row."""

import sys
from pathlib import Path

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import require_database_url


USER_PROFILES = [
    ("USER-001", "Kevin Chen"),
    ("USER-002", "Jack Ma"),
]


def main() -> None:
    with psycopg.connect(require_database_url()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                user_name TEXT NOT NULL
            )
        """)
        conn.cursor().executemany(
            """
            INSERT INTO user_profiles (user_id, user_name)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET user_name = EXCLUDED.user_name
            """,
            USER_PROFILES,
        )

    print(f"Inserted {len(USER_PROFILES)} user_profiles row(s).")


if __name__ == "__main__":
    main()

# command to verify the inserted data:
# python -c 'import psycopg; from config import require_database_url; conn=psycopg.connect(require_database_url()); cur=conn.execute("SELECT * FROM user_profiles ORDER BY user_id"); print(*cur.fetchall(), sep="\n"); conn.close()'

"""Environment-backed application configuration."""

import os

from dotenv import load_dotenv


load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

GEMINI_ROUTER_MODEL = os.getenv("GEMINI_ROUTER_MODEL", "gemini-3.5-flash").strip()
GEMINI_EXTRACTOR_MODEL = os.getenv("GEMINI_EXTRACTOR_MODEL", "gemini-3.5-flash").strip()
GEMINI_ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-3.5-flash").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()

BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
).strip()
BEDROCK_ROUTER_MODEL_ID = os.getenv("BEDROCK_ROUTER_MODEL_ID", BEDROCK_MODEL_ID).strip()
BEDROCK_EXTRACTOR_MODEL_ID = os.getenv("BEDROCK_EXTRACTOR_MODEL_ID", BEDROCK_MODEL_ID).strip()
BEDROCK_ANSWER_MODEL_ID = os.getenv("BEDROCK_ANSWER_MODEL_ID", BEDROCK_MODEL_ID).strip()
BEDROCK_EMBEDDING_MODEL_ID = os.getenv(
    "BEDROCK_EMBEDDING_MODEL_ID", "us.cohere.embed-v4:0"
).strip()
BEDROCK_EMBEDDING_DIMENSION = int(os.getenv("BEDROCK_EMBEDDING_DIMENSION", "1536"))
BEDROCK_REGION = (
    os.getenv("BEDROCK_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "us-east-1"
).strip()

DATABASE_BACKEND = os.getenv("DATABASE_BACKEND", "supabase").strip().lower()


def require_database_url() -> str:
    """Return the DSN for the selected database backend."""
    if DATABASE_BACKEND == "rds":
        database_url = os.getenv("RDS_DATABASE_URL")
        if not database_url:
            required = {
                "host": os.getenv("RDS_HOST"),
                "dbname": os.getenv("RDS_DB_NAME"),
                "user": os.getenv("RDS_USER"),
                "password": os.getenv("RDS_PASSWORD"),
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                names = ", ".join(
                    {
                        "host": "RDS_HOST",
                        "dbname": "RDS_DB_NAME",
                        "user": "RDS_USER",
                        "password": "RDS_PASSWORD",
                    }[name]
                    for name in missing
                )
                raise RuntimeError(f"Missing RDS connection settings: {names}.")

            from psycopg.conninfo import make_conninfo

            database_url = make_conninfo(
                **required,
                port=os.getenv("RDS_PORT", "5432"),
                sslmode="require",
            )
        variable_name = "RDS_DATABASE_URL or RDS_HOST/RDS_DB_NAME/RDS_USER/RDS_PASSWORD"
    elif DATABASE_BACKEND == "supabase":
        database_url = os.getenv("SUPABASE_DATABASE_URL") or os.getenv("DATABASE_URL")
        variable_name = "SUPABASE_DATABASE_URL"
    else:
        raise RuntimeError(
            f"Unsupported DATABASE_BACKEND: {DATABASE_BACKEND}. Choose 'rds' or 'supabase'."
        )

    if not database_url:
        raise RuntimeError(
            f"Missing {variable_name}. Add the PostgreSQL connection string to the .env file."
        )
    return database_url


def require_gemini_api_key() -> str:
    """Return the configured Gemini API key or raise a clear configuration error."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to the .env file.")
    return api_key

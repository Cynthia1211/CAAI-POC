"""Embedding adapter for the configured language model provider."""

import json
from typing import Any

from config import (
    BEDROCK_EMBEDDING_DIMENSION,
    BEDROCK_EMBEDDING_MODEL_ID,
    BEDROCK_REGION,
    LLM_PROVIDER,
)


def embed_text(text: str, task_type: str) -> list[float]:
    """Embed text with the configured provider and retrieval task type."""
    if LLM_PROVIDER == "bedrock":
        return _embed_with_bedrock(text, task_type)
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Choose 'bedrock'.")


def _embed_with_bedrock(text: str, task_type: str) -> list[float]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Bedrock requires boto3. Install dependencies from requirements.txt.") from exc

    model_id = BEDROCK_EMBEDDING_MODEL_ID
    if "titan-embed-text-v2" in model_id:
        # Titan Text Embeddings V2 uses a native inputText request and does not
        # distinguish query and document embeddings in its payload.
        request_body = json.dumps(
            {
                "inputText": text,
                "dimensions": BEDROCK_EMBEDDING_DIMENSION,
                "normalize": True,
            }
        )
    elif "cohere.embed-v4" in model_id:
        input_type = (
            "search_query"
            if task_type in {"RETRIEVAL_QUERY", "search_query"}
            else "search_document"
        )
        request_body = json.dumps(
            {
                "texts": [text],
                "input_type": input_type,
                "embedding_types": ["float"],
                "output_dimension": BEDROCK_EMBEDDING_DIMENSION,
            }
        )
    else:
        raise ValueError(
            f"Unsupported Bedrock embedding model format: {model_id}. "
            "Supported models: Titan Text Embeddings V2 and Cohere Embed v4."
        )

    response = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION).invoke_model(
        modelId=model_id,
        body=request_body,
        accept="application/json",
        contentType="application/json",
    )
    payload: dict[str, Any] = json.loads(response["body"].read())
    if "titan-embed-text-v2" in model_id:
        return payload["embedding"]

    embeddings = payload["embeddings"]
    if isinstance(embeddings, dict):
        embeddings = embeddings["float"]
    return embeddings[0]

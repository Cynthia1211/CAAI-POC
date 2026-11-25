"""Bedrock text generation and JSON response parsing."""

import json
from typing import Any


def generate_text(
    model_id: str,
    prompt: str,
    region_name: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0,
) -> str:
    """Generate text with Amazon Bedrock's Converse Runtime API."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Bedrock requires boto3. Install dependencies from requirements.txt.") from exc
    if not region_name:
        raise ValueError("region_name is required for Bedrock.")

    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"temperature": temperature},
    }
    if system_instruction:
        kwargs["system"] = [{"text": system_instruction}]

    response = boto3.client("bedrock-runtime", region_name=region_name).converse(**kwargs)
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    return "\n".join(block["text"] for block in blocks if "text" in block).strip()


def parse_json_object(response_text: str) -> dict[str, Any]:
    """Parse JSON from a model response, allowing fences or a short preamble."""
    text = response_text.strip()
    if not text:
        raise ValueError("The model returned an empty response; expected a JSON object.")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("The model response did not contain a valid JSON object.")

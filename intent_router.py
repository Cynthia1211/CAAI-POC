"""Classify a user question into the retrieval intents supported by the app."""

from typing import Literal

from pydantic import BaseModel

from config import (
    BEDROCK_REGION,
    BEDROCK_ROUTER_MODEL_ID,
    LLM_PROVIDER,
)
from llm_provider import generate_text, request_validated_json


Intent = Literal["personal", "calendar", "health", "general"]


class IntentClassification(BaseModel):
    """Validated single-category classification for a user question."""

    intent: Intent
    reason: str


def _generate_classification(prompt: str) -> str:
    if LLM_PROVIDER == "bedrock":
        return generate_text(
            model_id=BEDROCK_ROUTER_MODEL_ID,
            region_name=BEDROCK_REGION,
            prompt=prompt,
            temperature=0,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Choose 'bedrock'.")


def classify_intent(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> IntentClassification:
    """Classify the request in the context of the recent conversation."""
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in (history or [])
    ) or "(no prior conversation)"
    prompt = f"""
Classify the user question for a personal assistant retrieval workflow.
Recent conversation history (oldest first):
{history_text}
Return one JSON object with exactly these fields:
{{"intent": "calendar", "reason": "short explanation"}}

Choose exactly one intent:
- personal: the user asks about their own profile, family, preferences, or personal records.
- calendar: the user asks to view, create, change, or delete a calendar event or reminder. When the last assistant reply asks to confirm a calendar action or choose between calendar events, keep short confirmations and selections (yes, go ahead, the second one, or equivalent phrases in the user’s language) in calendar.
- health: the user asks general health, symptoms, disease, treatment, medication, or wellness questions that call for health information.
- general: ordinary conversation, greetings, writing help, follow-ups already answered by recent conversation history (such as repeating or summarizing the last response), or other questions that do not need new personal, calendar, or health retrieval.

Choose calendar for explicit calendar or reminder actions. Choose personal for questions about appointment records, attendees, family, and other personal information that do not ask to operate the calendar or reminders. Do not extract calendar action, event title, or event start/end times; those are handled separately.

Use the current question and recent conversation together to preserve intent continuity and resolve follow-up references. If the current question clearly starts a new topic, classify it normally.

Return valid JSON only.

Question: {query}
""".strip()
    return request_validated_json(
        prompt,
        _generate_classification,
        IntentClassification,
        "\nReturn valid JSON only. Use only the intent and reason fields.",
        "The intent classifier did not return a valid classification.",
    )

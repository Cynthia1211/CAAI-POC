"""Classify a user question into the retrieval intents supported by the app."""

from typing import Literal

from pydantic import BaseModel, ValidationError

from config import (
    BEDROCK_REGION,
    BEDROCK_ROUTER_MODEL_ID,
    GEMINI_ROUTER_MODEL,
    LLM_PROVIDER,
    require_gemini_api_key,
)
from llm_provider import generate_text, parse_json_object


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
    if LLM_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=require_gemini_api_key())
        response = client.models.generate_content(
            model=GEMINI_ROUTER_MODEL,
            contents=prompt,
        )
        if not response.text:
            raise ValueError("The intent classifier returned an empty response.")
        return response.text
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Choose 'gemini' or 'bedrock'.")


def classify_intent(query: str) -> IntentClassification:
    """Classify a question against the tools currently represented in the project."""
    prompt = f"""
Classify the user question for a personal assistant retrieval workflow.
Return one JSON object with exactly these fields:
{{"intent": "personal", "reason": "short explanation"}}

Choose exactly one intent:
- personal: the user asks about their own profile, family, appointments, calendar events, preferences, or personal records.
- calendar: the user asks to view, create, change, or delete a calendar event or reminder.
- health: the user asks general health, symptoms, disease, treatment, medication, or wellness questions that call for health information.
- general: ordinary conversation, greetings, writing help, or other questions that do not need personal or health retrieval.

Choose calendar for explicit calendar or reminder actions, such as checking upcoming calendar events or creating a reminder. Choose personal for questions about appointment records, attendees, family, and other personal information that do not ask to operate the calendar or reminders.

Return valid JSON only.

Question: {query}
""".strip()
    last_error: Exception | None = None
    for attempt in range(2):
        retry = "\nReturn valid JSON only. The intent must be personal, calendar, health, or general." if attempt else ""
        try:
            return IntentClassification.model_validate(
                parse_json_object(_generate_classification(prompt + retry))
            )
        except (ValueError, ValidationError) as exc:
            last_error = exc
    raise ValueError("The intent classifier did not return a valid classification.") from last_error

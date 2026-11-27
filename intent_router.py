"""Classify a user question into the retrieval intents supported by the app."""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError

from config import (
    BEDROCK_REGION,
    BEDROCK_ROUTER_MODEL_ID,
    LLM_PROVIDER,
)
from llm_provider import generate_text, parse_json_object


Intent = Literal["personal", "calendar", "health", "general"]


class IntentClassification(BaseModel):
    """Validated single-category classification for a user question."""

    intent: Intent
    reason: str
    date_from: str | None = None
    date_to: str | None = None
    time_zone: str = "America/Toronto"


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
    default_time_zone: str = "America/Toronto",
) -> IntentClassification:
    """Classify intent and extract generic date filters using the local date."""
    today = datetime.now(ZoneInfo(default_time_zone)).date()
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in (history or [])
    ) or "(no prior conversation)"
    prompt = f"""
Classify the user question for a personal assistant retrieval workflow.
Reference date (today): {today.isoformat()}
Default time zone: {default_time_zone}
Recent conversation history (oldest first):
{history_text}
Return one JSON object with exactly these fields:
{{"intent": "calendar", "reason": "short explanation", "date_from": null, "date_to": null, "time_zone": "{default_time_zone}"}}

Choose exactly one intent:
- personal: the user asks about their own profile, family, preferences, or personal records.
- calendar: the user asks to view, create, change, or delete a calendar event or reminder.
- health: the user asks general health, symptoms, disease, treatment, medication, or wellness questions that call for health information.
- general: ordinary conversation, greetings, writing help, follow-ups already answered by recent conversation history (such as repeating or summarizing the last response), or other questions that do not need new personal, calendar, or health retrieval.

Extract date filters as inclusive YYYY-MM-DD values in date_from and date_to. Convert relative dates and time spans—including today, tomorrow, a specific day, week, or month, and next N days, next N weeks, or next N months—using Reference date (today) and Default time zone above. For a single requested date, use the same date for both fields. Set time_zone to an IANA zone explicitly requested by the user, otherwise use the default. Leave both dates null when no date is clearly stated or implied.

Choose calendar for explicit calendar or reminder actions. Choose personal for questions about appointment records, attendees, family, and other personal information that do not ask to operate the calendar or reminders. Do not extract calendar action, event title, or event start/end times; those are handled separately.

Use the current question and recent conversation together to preserve intent continuity and resolve follow-up references. If the current question clearly starts a new topic, classify it normally.

Return valid JSON only.

Question: {query}
""".strip()
    last_error: Exception | None = None
    for attempt in range(2):
        retry = "\nReturn valid JSON only. Use only intent, reason, date_from, date_to, and time_zone fields." if attempt else ""
        try:
            payload = parse_json_object(_generate_classification(prompt + retry))
            payload.setdefault("time_zone", default_time_zone)
            classification = IntentClassification.model_validate(payload)
            return classification
        except (ValueError, ValidationError) as exc:
            last_error = exc
    raise ValueError("The intent classifier did not return a valid classification.") from last_error

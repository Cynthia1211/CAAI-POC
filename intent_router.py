"""Classify a user question into the retrieval intents supported by the app."""

import calendar
import re
from datetime import date
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
CalendarAction = Literal["view", "create", "delete"]


class IntentClassification(BaseModel):
    """Validated single-category classification for a user question."""

    intent: Intent
    reason: str
    calendar_action: CalendarAction | None = None
    summary: str | None = None
    start_iso: str | None = None
    end_iso: str | None = None
    time_zone: str = "America/Toronto"
    date_from: str | None = None
    date_to: str | None = None


_MONTH_COUNTS = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _next_months_window(today: date, month_count: int) -> tuple[str, str]:
    """Return an inclusive window from today through the same day N months later."""
    month_index = today.year * 12 + today.month - 1 + month_count
    target_year, zero_based_month = divmod(month_index, 12)
    target_month = zero_based_month + 1
    target_day = min(today.day, calendar.monthrange(target_year, target_month)[1])
    end_date = date(target_year, target_month, target_day)
    return today.isoformat(), end_date.isoformat()


def _apply_relative_view_window(
    query: str,
    classification: IntentClassification,
    today: date,
) -> IntentClassification:
    """Deterministically resolve common 'next N months' calendar view ranges."""
    if classification.intent != "calendar" or classification.calendar_action != "view":
        return classification
    match = re.search(
        r"\bnext\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b",
        query.casefold(),
    )
    if not match:
        return classification
    value = match.group(1)
    month_count = int(value) if value.isdigit() else _MONTH_COUNTS[value]
    date_from, date_to = _next_months_window(today, month_count)
    return classification.model_copy(update={"date_from": date_from, "date_to": date_to})


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


def classify_intent(
    query: str,
    reference_date: str | None = None,
) -> IntentClassification:
    """Classify a question against the tools currently represented in the project."""
    today = date.fromisoformat(reference_date) if reference_date else date.today()
    prompt = f"""
Classify the user question for a personal assistant retrieval workflow.
Reference date (today): {today.isoformat()}
Return one JSON object with exactly these fields:
{{"intent": "calendar", 
"reason": "short explanation", 
"calendar_action": "view", 
"summary": null, 
"start_iso": null, 
"end_iso": null, 
"time_zone": "America/Toronto", 
"date_from": null, 
"date_to": null}}

Choose exactly one intent:
- personal: the user asks about their own profile, family, appointments, calendar events, preferences, or personal records.
- calendar: the user asks to view, create, change, or delete a calendar event or reminder. Set calendar_action to view, create, or delete. For creates, resolve relative dates such as today, tomorrow, or next Friday using the reference date above. Extract start_iso and end_iso as local wall-clock date-times in YYYY-MM-DDTHH:MM:SS format without a UTC offset. Set time_zone to an IANA time zone from the user's request, or America/Toronto if none is given. Do not convert local times to UTC. Leave a field null if the user did not provide enough information to determine it; do not invent a time or duration. For views with a requested date interval, resolve date_from and date_to as inclusive YYYY-MM-DD dates using the reference date above. For "the next N months", use today as date_from and the same calendar day N months later as date_to (clamp to the last day of that month if needed). Leave both null when no interval is requested. For view/delete, set summary, start_iso, and end_iso to null.
- health: the user asks general health, symptoms, disease, treatment, medication, or wellness questions that call for health information.
- general: ordinary conversation, greetings, writing help, or other questions that do not need personal or health retrieval.

Choose calendar for explicit calendar or reminder actions, such as viewing upcoming calendar events, creating a reminder, or deleting an event. Choose personal for questions about appointment records, attendees, family, and other personal information that do not ask to operate the calendar or reminders.

Return valid JSON only.

Question: {query}
""".strip()
    last_error: Exception | None = None
    for attempt in range(2):
        retry = "\nReturn valid JSON only. The intent must be personal, calendar, health, or general; calendar_action must be view, create, or delete for calendar intents." if attempt else ""
        try:
            classification = IntentClassification.model_validate(
                parse_json_object(_generate_classification(prompt + retry))
            )
            return _apply_relative_view_window(query, classification, today)
        except (ValueError, ValidationError) as exc:
            last_error = exc
    raise ValueError("The intent classifier did not return a valid classification.") from last_error

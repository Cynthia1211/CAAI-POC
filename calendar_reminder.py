"""Google Calendar operations for an already authorized account."""

import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from chunk_builder import build_chunks
from config import BEDROCK_EXTRACTOR_MODEL_ID, BEDROCK_REGION, LLM_PROVIDER
from llm_provider import generate_text, parse_json_object


router = APIRouter()
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
BASE_DIR = Path(__file__).resolve().parent / "test" / "googleCalendar"
CONFIG_PATH = BASE_DIR / ".env.json"
load_dotenv(Path(__file__).resolve().parent / ".env")
SESSION_USER_ID = "demo_user"
TOKEN_STORE: dict[str, dict[str, Any]] = {}


CalendarAction = Literal["view", "create", "delete"]


class CalendarOperation(BaseModel):
    """Calendar-specific action and event details extracted for execution."""

    action: CalendarAction
    summary: str | None = None
    start_iso: str | None = None
    end_iso: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    time_zone: str = "America/Toronto"


def extract_calendar_operation(
    query: str,
    history: list[dict[str, str]] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    time_zone: str = "America/Toronto",
) -> CalendarOperation:
    """Extract action and event details using the calendar-specific context."""
    reference_date = datetime.now(ZoneInfo(time_zone)).date().isoformat()
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in (history or [])
    ) or "(no prior conversation)"
    prompt = f"""
Extract a calendar operation from the current request and recent conversation.
Recent conversation (oldest first):
{history_text}
Current request: {query}
Known date_from: {date_from or 'none'}
Known date_to: {date_to or 'none'}
Time zone: {time_zone}
Reference date in that time zone: {reference_date}

Return exactly one JSON object with these fields:
{{"action":"view","summary":null,"start_iso":null,"end_iso":null,"date_from":null,"date_to":null,"time_zone":"{time_zone}"}}

Choose action view, create, or delete. Continue an unfinished calendar action when the current request supplies missing details; merge title, date, time, and time zone from relevant history. If the user changes topic or action, follow the current request. For creates, return local wall-clock start_iso/end_iso in YYYY-MM-DDTHH:MM:SS format without UTC offset. Resolve relative dates using the reference date above. Use known date_from/date_to as authoritative date hints. Do not invent missing titles, dates, or times. For view, use known date bounds if present. Leave irrelevant fields null.
""".strip()
    if LLM_PROVIDER != "bedrock":
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Choose 'bedrock'.")

    last_error: Exception | None = None
    for attempt in range(2):
        retry = (
            "\nReturn valid JSON only with action, summary, start_iso, end_iso, date_from, date_to, and time_zone."
            if attempt
            else ""
        )
        try:
            output = generate_text(
                model_id=BEDROCK_EXTRACTOR_MODEL_ID,
                region_name=BEDROCK_REGION,
                prompt=prompt + retry,
                temperature=0,
            )
            parsed = CalendarOperation.model_validate(parse_json_object(output))
            updates: dict[str, str | None] = {"time_zone": time_zone}
            if date_from is not None:
                updates["date_from"] = date_from
            if date_to is not None:
                updates["date_to"] = date_to
            return parsed.model_copy(update=updates)
        except (ValueError, ValidationError) as exc:
            last_error = exc
    raise ValueError("The calendar extractor did not return a valid operation.") from last_error


def set_calendar_token(token: str, refresh_token: str | None) -> None:
    """Accept tokens produced by the OAuth test flow in test/googleCalendar/app.py."""
    TOKEN_STORE[SESSION_USER_ID] = {"token": token, "refresh_token": refresh_token}


def _config() -> dict[str, str]:
    """Read the OAuth client credentials used to construct API credentials."""
    file_config: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        file_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    values = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID") or file_config.get("client_id"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET") or file_config.get("client_secret"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Google Calendar client is not configured. Set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET (missing " + ", ".join(missing) + ")."
        )
    return values  # type: ignore[return-value]


def _get_calendar_service():
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Install Google Calendar dependencies with: pip install -r requirements.txt") from exc
    token_data = TOKEN_STORE.get(SESSION_USER_ID)
    if not token_data:
        raise HTTPException(
            status_code=401,
            detail="Google Calendar is not authorized. Complete the OAuth login first.",
        )
    config = _config()
    credentials = Credentials(
        token=token_data["token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        scopes=SCOPES,
    )
    return build("calendar", "v3", credentials=credentials)


def _normalize_calendar_datetime(
    value: str,
    time_zone: str,
    interpret_as_local_time: bool = False,
) -> str:
    """Interpret naive inputs in the selected local zone; convert aware inputs to it."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    zone = ZoneInfo(time_zone)
    if interpret_as_local_time:
        parsed = parsed.replace(tzinfo=zone)
    elif parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    else:
        parsed = parsed.astimezone(zone)
    return parsed.isoformat()


def _event_time_value(value: dict[str, Any], fallback: str) -> str:
    return value.get("dateTime") or value.get("date") or fallback


def view_calendar_events(
    max_results: int = 50,
    date_from: str | None = None,
    date_to: str | None = None,
    time_zone: str = "America/Toronto",
) -> list[dict[str, str]]:
    """List events in a date window, including each Google Calendar event ID."""
    zone = ZoneInfo(time_zone)
    if date_from:
        time_min = datetime.combine(date.fromisoformat(date_from), time.min, tzinfo=zone)
    else:
        time_min = datetime.now(zone)
    params: dict[str, Any] = {
        "calendarId": "primary",
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
        "timeMin": time_min.isoformat(),
    }
    if date_to:
        # Calendar API timeMax is exclusive; add one day to include date_to itself.
        time_max = datetime.combine(
            date.fromisoformat(date_to) + timedelta(days=1), time.min, tzinfo=zone
        )
        params["timeMax"] = time_max.isoformat()
    response = _get_calendar_service().events().list(**params).execute()
    return [
        {
            "id": item["id"],
            "summary": item.get("summary", "(Untitled)"),
            "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date", ""),
            "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date", ""),
            "time_zone": item.get("start", {}).get("timeZone") or "America/Toronto",
        }
        for item in response.get("items", [])
    ]


def add_calendar_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    time_zone: str = "America/Toronto",
    interpret_as_local_time: bool = False,
) -> dict[str, Any]:
    """Create an event, fetch it again by ID, and report creation/verification status."""
    created: dict[str, Any] | None = None
    try:
        normalized_start = _normalize_calendar_datetime(start_iso, time_zone, interpret_as_local_time)
        normalized_end = _normalize_calendar_datetime(end_iso, time_zone, interpret_as_local_time)
        service = _get_calendar_service()
        created = service.events().insert(
            calendarId="primary",
            body={
                "summary": summary,
                "start": {"dateTime": normalized_start, "timeZone": time_zone},
                "end": {"dateTime": normalized_end, "timeZone": time_zone},
            },
        ).execute()
    except Exception as exc:
        return {
            "success": False,
            "created": False,
            "verified": False,
            "reason": f"Event creation failed: {type(exc).__name__}: {exc}",
            "event": None,
        }

    event_id = created.get("id")
    if not event_id:
        return {
            "success": False,
            "created": True,
            "verified": False,
            "reason": "Google Calendar returned a create response without an event ID, so the event could not be verified.",
            "event": created,
        }

    try:
        verified = service.events().get(calendarId="primary", eventId=event_id).execute()
    except Exception as exc:
        return {
            "success": False,
            "created": True,
            "verified": False,
            "reason": f"Google Calendar created event {event_id}, but the follow-up lookup failed: {type(exc).__name__}: {exc}",
            "event": _calendar_event_record(created, summary, normalized_start, normalized_end, time_zone),
        }

    return {
        "success": True,
        "created": True,
        "verified": True,
        "reason": "Google Calendar created the event and the follow-up lookup confirmed it.",
        "event": _calendar_event_record(verified, summary, normalized_start, normalized_end, time_zone),
    }


def _calendar_event_record(
    event: dict[str, Any],
    summary_fallback: str,
    start_fallback: str,
    end_fallback: str,
    time_zone_fallback: str,
) -> dict[str, str]:
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id", ""),
        "summary": event.get("summary", summary_fallback),
        "start": _event_time_value(start, start_fallback),
        "end": _event_time_value(end, end_fallback),
        "time_zone": start.get("timeZone", time_zone_fallback),
        "status": event.get("status", "confirmed"),
        "html_link": event.get("htmlLink", ""),
    }


def update_calendar_event(
    event_id: str,
    *,
    summary: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    time_zone: str = "America/Toronto",
) -> dict[str, str]:
    """Update an event's title and/or time, returning its Google Calendar ID."""
    if (start_iso is None) != (end_iso is None):
        raise ValueError("start_iso and end_iso must be provided together.")
    body: dict[str, Any] = {}
    if summary is not None:
        body["summary"] = summary
    if start_iso is not None and end_iso is not None:
        body["start"] = {
            "dateTime": _normalize_calendar_datetime(start_iso, time_zone),
            "timeZone": time_zone,
        }
        body["end"] = {
            "dateTime": _normalize_calendar_datetime(end_iso, time_zone),
            "timeZone": time_zone,
        }
    if not body:
        raise ValueError("Provide a summary or a start/end time to update.")
    updated = _get_calendar_service().events().patch(
        calendarId="primary", eventId=event_id, body=body
    ).execute()
    return {
        "id": updated["id"],
        "summary": updated.get("summary", ""),
        "start": _event_time_value(updated.get("start", {}), start_iso or ""),
        "end": _event_time_value(updated.get("end", {}), end_iso or ""),
        "time_zone": updated.get("start", {}).get("timeZone", time_zone),
    }


def delete_calendar_event(event_id: str) -> dict[str, Any]:
    """Delete a calendar event by its Google Calendar ID."""
    _get_calendar_service().events().delete(calendarId="primary", eventId=event_id).execute()
    return {"deleted": True, "id": event_id}


@router.get("/api/events")
def api_view_events():
    return view_calendar_events()


class EventCreate(BaseModel):
    summary: str
    start_iso: str
    end_iso: str
    time_zone: str = "America/Toronto"


class EventUpdate(BaseModel):
    summary: str | None = None
    start_iso: str | None = None
    end_iso: str | None = None
    time_zone: str = "America/Toronto"


@router.post("/api/events")
def api_add_event(payload: EventCreate):
    return add_calendar_event(payload.summary, payload.start_iso, payload.end_iso, payload.time_zone)


@router.patch("/api/events/{event_id}")
def api_update_event(event_id: str, payload: EventUpdate):
    try:
        return update_calendar_event(event_id, **payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/api/events/{event_id}")
def api_delete_event(event_id: str):
    return delete_calendar_event(event_id)


def format_calendar_events(events: list[dict[str, str]]) -> str:
    """Format Google Calendar events as readable grounded context."""
    return "\n\n".join(
        "\n".join((
            f"Desc: {event.get('summary', '(Untitled)')}",
            f"Date from: {_format_calendar_time(event.get('start', ''), event.get('time_zone', 'America/Toronto'))}",
            f"Date to: {_format_calendar_time(event.get('end', ''), event.get('time_zone', 'America/Toronto'))}",
        ))
        for event in events
    )


def _format_calendar_time(value: str, time_zone: str) -> str:
    """Render event timestamps in the event's local time zone for grounding."""
    if not value or "T" not in value:
        return value
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    zone = ZoneInfo(time_zone)
    local = parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)
    return local.strftime("%Y-%m-%d %I:%M %p %Z")


def calendar_events_to_chunks(
    events: list[dict[str, str]],
    result_note: str | None = None,
) -> list[dict[str, Any]]:
    """Format events and wrap them in the shared Google Calendar chunk shape."""
    content = format_calendar_events(events)
    if result_note:
        content = f"{result_note}\n\n{content}" if content else result_note
    if not content:
        content = "No upcoming calendar events."
    return build_chunks(
        query_results=[{
            "source": "Google Calendar",
            "records": [{"fields": {"content": content}}],
        }],
        chunk_prefix="Calendar events",
        chunk_name="Calendar events",
        empty_detail=content,
    )


def calendar_updated_to_chunk(result: dict[str, Any]) -> dict[str, Any]:
    """Build the creation-result chunk, including the event fetched by its ID."""
    success = bool(result.get("success"))
    lines = [
        f"Result: {'Success' if success else 'Failure'}",
        f"Reason: {result.get('reason', 'No result reason was returned.')}",
    ]
    event = result.get("event")
    if event:
        lines.extend([
            "",
            format_calendar_events([event]),
            f"ID: {event.get('id', 'N/A')}",
        ])
    content = "\n".join(lines)
    chunks = build_chunks(
        query_results=[{
            "source": "Google Calendar",
            "records": [{"fields": {"content": content}}],
        }],
        chunk_prefix="Calendar Updated",
        chunk_name="Calendar Updated",
        empty_detail=content,
    )
    return chunks[0]


def _run_calendar_view(operation: CalendarOperation) -> list[dict[str, Any]]:
    """Run a calendar lookup and return its grounded context chunk."""
    events = view_calendar_events(
        date_from=operation.date_from,
        date_to=operation.date_to,
        time_zone=operation.time_zone,
    )
    today = datetime.now(ZoneInfo(operation.time_zone)).date().isoformat()
    window_note = (
        f"Reference date: {today}. Calendar search window: "
        f"{operation.date_from or 'now'} through "
        f"{operation.date_to or 'no end date'} ({operation.time_zone})."
    )
    return calendar_events_to_chunks(events, result_note=window_note)


def _run_calendar_create(operation: CalendarOperation) -> list[dict[str, Any]]:
    """Create a calendar event when the required details are available."""
    if not operation.summary or not operation.start_iso or not operation.end_iso:
        return calendar_events_to_chunks(
            [], result_note="Please provide an event title, date, start time, and end time."
        )
    result = add_calendar_event(
        operation.summary,
        operation.start_iso,
        operation.end_iso,
        operation.time_zone,
        interpret_as_local_time=True,
    )
    return [calendar_updated_to_chunk(result)]


def _run_calendar_delete(operation: CalendarOperation) -> list[dict[str, Any]]:
    """Find and delete the requested calendar event, returning a result chunk."""
    events = view_calendar_events(
        date_from=operation.date_from,
        date_to=operation.date_to,
        time_zone=operation.time_zone,
    )
    if not events:
        return calendar_events_to_chunks(
            [], result_note="There are no upcoming events to delete."
        )

    target_event = events[0]
    if operation.summary:
        target_event = next(
            (
                event for event in events
                if event.get("summary", "").casefold() == operation.summary.casefold()
            ),
            {},
        )
    if not target_event:
        return calendar_events_to_chunks(
            [], result_note=f"No calendar event named '{operation.summary}' was found in that date range."
        )

    delete_calendar_event(target_event["id"])
    return calendar_events_to_chunks(
        [target_event], result_note="Event deleted successfully."
    )


def execute_calendar_operation(operation: CalendarOperation) -> list[dict[str, Any]]:
    """Dispatch a validated calendar operation to its execution function."""
    action_handlers = {
        "view": _run_calendar_view,
        "create": _run_calendar_create,
        "delete": _run_calendar_delete,
    }
    return action_handlers[operation.action](operation)

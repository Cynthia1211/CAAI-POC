"""Shared calendar extraction, time formatting, and operation dispatch."""

import json
import re
from datetime import datetime
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

from chunk_builder import build_chunks
from config import BEDROCK_EXTRACTOR_MODEL_ID, BEDROCK_REGION, DEFAULT_TIME_ZONE, LLM_PROVIDER
from llm_provider import generate_text, request_validated_json


# Operation model and provider interface

CalendarAction = Literal["view", "create", "delete"]


class CalendarCreateItem(BaseModel):
    summary: str | None = None
    start_iso: str | None = None
    end_iso: str | None = None
    time_zone: str | None = None


class CalendarOperation(BaseModel):
    """Calendar-specific action and event details extracted for execution."""

    events: list[CalendarCreateItem] = Field(default_factory=list)
    action: CalendarAction
    confirmed: bool = False
    event_id: str | None = None
    event_ids: list[str] = Field(default_factory=list)  # Explicit multi-event selection.
    summary: str | None = None
    start_iso: str | None = None
    end_iso: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    time_zone: str = DEFAULT_TIME_ZONE

    @field_validator("time_zone", mode="before")
    @classmethod
    def default_zone_if_empty(cls, value):
        return value or DEFAULT_TIME_ZONE


class CalendarProvider(Protocol):
    """A calendar backend supplies normalized event records and its chunk source."""

    SOURCE: str

    def view_calendar_events_by_date(
        self, *, date_from: str | None, date_to: str | None, time_zone: str, max_results: int | None = 50
    ) -> list[dict[str, str]]: ...

    def get_calendar_event_by_id(self, event_id: str) -> dict[str, str] | None: ...

    def add_calendar_event(
        self, summary: str, start_iso: str, end_iso: str, time_zone: str
    ) -> dict[str, Any]: ...

    def delete_calendar_event(self, event_id: str) -> dict[str, Any]: ...


# Operation extraction and dispatch

def extract_calendar_operation(
    query: str,
    history: list[dict[str, str]] | None = None,
    pending_ids: list[str] | None = None,
) -> CalendarOperation:
    """Extract action and event details using the calendar-specific context."""
    reference_instant = datetime.now(ZoneInfo(DEFAULT_TIME_ZONE))
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in (history or [])
    ) or "(no prior conversation)"
    prompt = f"""
Extract a calendar operation from the current request and recent conversation.
Recent conversation (oldest first):
{history_text}
Current request: {query}
Pending candidate IDs in order: {json.dumps(pending_ids or [], ensure_ascii=False)}
Default time zone: {DEFAULT_TIME_ZONE}
Reference instant: {reference_instant.isoformat()}

Return exactly one JSON object with these fields:
{{"action":"view","confirmed":false,"event_id":null,"event_ids":[],"events":[],"summary":null,"start_iso":null,"end_iso":null,"date_from":null,"date_to":null,"time_zone":"{DEFAULT_TIME_ZONE}"}}

Choose view, create, or delete. Use history to complete an unfinished request; follow the current request when the topic changes.
Resolve relative dates from the reference instant in the requested IANA zone (otherwise the default). Search dates are inclusive YYYY-MM-DD; for a single day use the same date for both bounds. Leave unknown details null.
For create, put every event, including a single event, in events with summary, start_iso, end_iso and optional time_zone. Use local YYYY-MM-DDTHH:MM:SS in the item's zone or the operation zone. For other actions leave events empty.
For delete, use top-level title (summary) and time filters. Preview first. Only confirm an explicit authorization to delete previously presented candidates. Resolve IDs or 1-based positions against the ordered pending list. Copy only the selected IDs: event_id for one, event_ids for several. Selecting without authorizing, an unclear target, or a bare yes to multiple candidates leaves confirmed=false. Delete all only when explicitly requested.
For example, after presenting candidates, "please delete the second one only" explicitly authorize deletion: set confirmed=true and event_id to the second pending ID. "the second one" only selects it, so set confirmed=false. "delete the first two" explicitly authorizes exactly the first two IDs.
""".strip()
    if LLM_PROVIDER != "bedrock":
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Choose 'bedrock'.")

    parsed = request_validated_json(
        prompt,
        lambda text: generate_text(
                model_id=BEDROCK_EXTRACTOR_MODEL_ID,
                region_name=BEDROCK_REGION,
                prompt=text,
                temperature=0,
        ),
        CalendarOperation,
        "\nReturn valid JSON only with action, confirmed, event_id, event_ids, events, summary, start_iso, end_iso, date_from, date_to, and time_zone.",
        "The calendar extractor did not return a valid operation.",
    )
    updates: dict[str, Any] = {}
    if pending_ids is None and not (history and history[-1].get("role") == "assistant"):
        pending_ids = []
    if parsed.action == "delete" and pending_ids is not None:
        reply = query.strip().casefold().rstrip(".!?").strip()
        bare_confirmation = reply in {"yes", "ok", "go ahead"}
        if not pending_ids:
            updates.update(confirmed=False, event_id=None, event_ids=[])
        elif bare_confirmation:
            if len(pending_ids) == 1:
                updates.update(confirmed=True, event_id=pending_ids[0], event_ids=[])
            else:
                updates.update(confirmed=False, event_id=None, event_ids=pending_ids)
        else:
            # Recognize only complete, unambiguous deletion commands after a preview.
            english = re.fullmatch(
                r"(?:yes[, ]+)?(?:please\s+)?(?:delete|remove)\s+(?:the\s+)?(first|second|third)\s+(?:one|item|event)(?:\s+only)?(?:\s+please)?",
                reply,
            )
            positions = {"first": 0, "second": 1, "third": 2}
            match = english
            if match and positions[match.group(1)] < len(pending_ids):
                updates.update(confirmed=True, event_id=pending_ids[positions[match.group(1)]], event_ids=[])
    return parsed.model_copy(update=updates)


# Time and event formatting

def _parse_to_zone(value: str, time_zone: str) -> datetime:
    """Parse a timestamp and return it in the requested time zone."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    zone = ZoneInfo(time_zone)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    else:
        parsed = parsed.astimezone(zone)
    return parsed


def _normalize_calendar_datetime(
    value: str,
    time_zone: str,
) -> str:
    """Interpret naive inputs in the selected local zone; convert aware inputs to it."""
    return _parse_to_zone(value, time_zone).isoformat()


def _format_calendar_time(value: str, time_zone: str) -> str:
    """Render event timestamps in the event's local time zone for grounding."""
    if not value or "T" not in value:
        return value
    return _parse_to_zone(value, time_zone).strftime("%Y-%m-%d %I:%M %p %Z")


def format_calendar_events(events: list[dict[str, str]]) -> str:
    """Format calendar events as readable grounded context."""
    return "\n\n".join(
        "\n".join((
            f"Desc: {event.get('summary', '(Untitled)')}",
            f"Date from: {_format_calendar_time(event.get('start', ''), event.get('time_zone', DEFAULT_TIME_ZONE))}",
            f"Date to: {_format_calendar_time(event.get('end', ''), event.get('time_zone', DEFAULT_TIME_ZONE))}",
        ))
        for event in events
    )


# View and create operations

def _run_calendar_view(
    operation: CalendarOperation, provider: CalendarProvider
) -> list[dict[str, Any]]:
    """Run a calendar lookup and return its grounded context chunk."""
    events = provider.view_calendar_events_by_date(
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
    event_details = format_calendar_events(events) or "No calendar events found in this date range."
    content = f"{window_note}\n\n{event_details}"
    return build_chunks(
        query_results=[{"source": provider.SOURCE, "records": [{"fields": {"content": content}}]}],
        chunk_prefix="Calendar View",
        chunk_name="Calendar View",
        empty_detail=content,
    )


def _run_calendar_create(
    operation: CalendarOperation, provider: CalendarProvider
) -> list[dict[str, Any]]:
    """Validate the batch before writing, then report each attempt independently."""
    items = operation.events
    missing = []
    for index, item in enumerate(items, 1):
        fields = [label for label, value in (
            ("title", item.summary), ("start date/time", item.start_iso), ("end date/time", item.end_iso)
        ) if not value or not value.strip()]
        if fields:
            missing.append(f"{index}. {item.summary or '(untitled)'}: missing {', '.join(fields)}")
    if not items or missing:
        content = "No events were submitted. Please provide an event title, date, start time, and end time for each event."
        if missing:
            content += "\n" + "\n".join(missing)
    else:
        reports = []
        for index, item in enumerate(items, 1):
            zone = item.time_zone or operation.time_zone
            try:
                result = provider.add_calendar_event(item.summary, item.start_iso, item.end_iso, zone)
            except Exception as exc:
                result = {"success": False, "reason": f"{type(exc).__name__}: {exc}"}
            event = result.get("event") or {}
            reports.append("\n".join((
                f"{index}. {item.summary}",
                f"Result: {'Success' if result.get('success') else 'Failure'}",
                f"Reason: {result.get('reason') or 'No reason returned.'}",
                f"Date from: {event.get('start') or item.start_iso}",
                f"Date to: {event.get('end') or item.end_iso}",
                f"Time zone: {event.get('time_zone') or zone}",
                f"ID: {event.get('id') or 'N/A'}",
            )))
        content = "\n\n".join(reports)
    return build_chunks(
        query_results=[{"source": provider.SOURCE, "records": [{"fields": {"content": content}}]}],
        chunk_prefix="Calendar Add",
        chunk_name="Calendar Add",
        empty_detail=content,
    )


# Delete matching and confirmation

def _matches_event_time(value: str, requested: str, zone: str) -> bool:
    if not value:
        return False
    actual = _parse_to_zone(value, zone)
    expected = _parse_to_zone(requested, zone)
    if len(requested) == 10:
        return actual.date() == expected.date()
    return len(value) > 10 and actual.timestamp() == expected.timestamp()


def _find_delete_matches(operation: CalendarOperation, provider: CalendarProvider) -> list[dict[str, str]]:
    start_day = (
        _parse_to_zone(operation.start_iso, operation.time_zone).date().isoformat()
        if operation.start_iso else None
    )
    end_day = operation.date_to or start_day
    events = provider.view_calendar_events_by_date(
        max_results=None if end_day else 50,
        date_from=operation.date_from or start_day,
        date_to=end_day,
        time_zone=operation.time_zone,
    )
    matches = [
        event for event in events if event.get("id")
        and all(
            not requested or _matches_event_time(event.get(field, ""), requested, operation.time_zone)
            for field, requested in (("start", operation.start_iso), ("end", operation.end_iso))
        )
    ]
    if not operation.summary:
        return matches
    title = operation.summary.strip().casefold()
    exact = [e for e in matches if e.get("summary", "").casefold() == title]
    return exact or [e for e in matches if title in e.get("summary", "").casefold()]


def _delete_details(event: dict[str, str]) -> str:
    return f"{format_calendar_events([event])}\nID: {event['id']}"


def _run_calendar_delete(
    operation: CalendarOperation, provider: CalendarProvider, pending_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Preview candidates, then delete only explicitly confirmed targets."""
    selected = list(dict.fromkeys(([operation.event_id] if operation.event_id else []) + operation.event_ids))
    confirmed = operation.confirmed and (pending_ids is None or bool(pending_ids))
    if selected and pending_ids is not None and not set(selected).issubset(pending_ids):
        matches = []
    elif selected:
        found = [provider.get_calendar_event_by_id(event_id) for event_id in selected]
        matches = [event for event in found if event]
        if len(matches) != len(selected):
            matches = []  # Re-select the group if any requested ID has disappeared.
    else:
        matches = _find_delete_matches(operation, provider)
        if confirmed and pending_ids is not None:
            matches = [event for event in matches if event["id"] in pending_ids]

    has_filters = any((operation.summary, operation.start_iso, operation.end_iso, operation.date_from, operation.date_to))
    target_resolved = bool(selected) or (len(matches) == 1 and has_filters)
    pending = []
    if not matches:
        content = (
            f"Status: No Matches Found\nNo matching calendar event was found. "
            f"Title: {operation.summary or 'unspecified'}. Search window: "
            f"{operation.date_from or operation.start_iso or 'now'} through "
            f"{operation.date_to or operation.end_iso or 'no end date'} ({operation.time_zone}). "
            "Please provide the event name, specific date, or time range."
        )
    elif confirmed and target_resolved:
        results = []
        for event in matches:
            try:
                result = provider.delete_calendar_event(event["id"])
                deleted = result.get("deleted")
            except Exception as exc:
                results.append(f"Result: Failure\nDeletion could not be verified ({type(exc).__name__}).\n{_delete_details(event)}")
                break
            status = "Result: Success\nEvent deleted successfully." if deleted else "Result: Failure\nEvent was not deleted."
            if result.get("reason"):
                status += f"\nReason: {result['reason']}"
            results.append(f"{status}\n{_delete_details(event)}")
            if not deleted:
                break
        remaining = matches[len(results):]
        if remaining:
            results.append("Not attempted: " + ", ".join(event["id"] for event in remaining))
        content = "\n\n".join(results)
    else:
        pending = [event["id"] for event in matches]
        if len(matches) == 1:
            note = "Status: Confirmation Required\nPlease confirm whether to delete this event."
        else:
            note = (
                "Status: Multiple Matches Found\nPlease specify which event(s) to delete, "
                "or confirm deletion of all listed events."
            )
        content = "\n\n".join([note, *map(_delete_details, matches)])

    chunks = build_chunks(
        query_results=[{"source": provider.SOURCE, "records": [{"fields": {"content": content}}]}],
        chunk_prefix="Calendar Delete",
        chunk_name="Calendar Delete",
        empty_detail=content,
    )
    if pending:
        chunks[0]["pending_delete_ids"] = pending
    chunks[0]["delete_receipt"] = content
    return chunks

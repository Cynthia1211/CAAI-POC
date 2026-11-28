"""Google Calendar operations for an already authorized account."""

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import DEFAULT_TIME_ZONE
from calendar_utils import _normalize_calendar_datetime


SOURCE = "Google Calendar"
router = APIRouter()
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
load_dotenv(Path(__file__).resolve().parent / ".env")
SESSION_USER_ID = "demo_user"
TOKEN_STORE: dict[str, dict[str, Any]] = {}


# OAuth and API client

def google_oauth_config() -> dict[str, str]:
    """Read the OAuth settings shared by login and event operations."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in the environment or root .env.")
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI") or "http://localhost:8000/auth/callback",
    }


def set_calendar_token(token: str, refresh_token: str | None) -> None:
    """Accept tokens produced by the OAuth test flow in test/googleCalendar/app.py."""
    TOKEN_STORE[SESSION_USER_ID] = {"token": token, "refresh_token": refresh_token}


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
    config = google_oauth_config()
    credentials = Credentials(
        token=token_data["token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        scopes=SCOPES,
    )
    return build("calendar", "v3", credentials=credentials)


# Google event serialization

def _event_time_value(value: dict[str, Any], fallback: str) -> str:
    return value.get("dateTime") or value.get("date") or fallback


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


# Event queries

def view_calendar_events_by_date(
    max_results: int | None = 50,
    date_from: str | None = None,
    date_to: str | None = None,
    time_zone: str = DEFAULT_TIME_ZONE,
) -> list[dict[str, str]]:
    """List events in a date window, including each Google Calendar event ID."""
    zone = ZoneInfo(time_zone)
    if date_from:
        time_min = datetime.combine(date.fromisoformat(date_from), time.min, tzinfo=zone)
    else:
        time_min = datetime.now(zone)
    params: dict[str, Any] = {
        "calendarId": "primary",
        "maxResults": 2500 if max_results is None else max_results,
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
    service = _get_calendar_service().events()
    events = []
    while True:
        response = service.list(**params).execute()
        events.extend(
            _calendar_event_record(item, "(Untitled)", "", "", time_zone)
            for item in response.get("items", [])
        )
        if max_results is not None or not response.get("nextPageToken"):
            return events
        params["pageToken"] = response["nextPageToken"]


def get_calendar_event_by_id(event_id: str) -> dict[str, str] | None:
    """Resolve a selected ID independently of the current list window."""
    from googleapiclient.errors import HttpError

    try:
        event = _get_calendar_service().events().get(calendarId="primary", eventId=event_id).execute()
    except HttpError as exc:
        if exc.resp.status in (404, 410):
            return None
        raise
    if event.get("status") == "cancelled":
        return None
    return _calendar_event_record(event, "(Untitled)", "", "", DEFAULT_TIME_ZONE)


# Event creation and deletion

def add_calendar_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    time_zone: str = DEFAULT_TIME_ZONE,
) -> dict[str, Any]:
    """Create an event, fetch it again by ID, and report creation/verification status."""
    created: dict[str, Any] | None = None
    try:
        normalized_start = _normalize_calendar_datetime(start_iso, time_zone)
        normalized_end = _normalize_calendar_datetime(end_iso, time_zone)
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
            "event": _calendar_event_record(created, summary, normalized_start, normalized_end, time_zone),
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


def delete_calendar_event(event_id: str) -> dict[str, Any]:
    """Delete an event and confirm that it is no longer active."""
    _get_calendar_service().events().delete(calendarId="primary", eventId=event_id).execute()
    try:
        remaining = get_calendar_event_by_id(event_id)
    except Exception as exc:
        return {"deleted": False, "id": event_id, "reason": f"Delete request accepted, but verification failed: {type(exc).__name__}: {exc}"}
    if remaining is not None:
        return {"deleted": False, "id": event_id, "reason": "The event is still active in Google Calendar after the delete request."}
    return {"deleted": True, "id": event_id, "reason": "The event is no longer active in Google Calendar."}


# HTTP request model and routes

class EventCreate(BaseModel):
    summary: str
    start_iso: str
    end_iso: str
    time_zone: str = DEFAULT_TIME_ZONE


@router.get("/api/events")
def api_view_events():
    return view_calendar_events_by_date()


@router.post("/api/events")
def api_add_event(payload: EventCreate):
    return add_calendar_event(payload.summary, payload.start_iso, payload.end_iso, payload.time_zone)


@router.delete("/api/events/{event_id}")
def api_delete_event(event_id: str):
    return delete_calendar_event(event_id)

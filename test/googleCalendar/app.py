"""Google OAuth test flow. Calendar CRUD lives in calendar_google.py."""

import sys
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calendar_google import SCOPES, google_oauth_config, set_calendar_token


CODE_VERIFIER_STORE: dict[str, str] = {}
auth_router = APIRouter()


def _build_flow():
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError as exc:
        raise RuntimeError("Install Google Calendar dependencies with: pip install -r requirements.txt") from exc
    config = google_oauth_config()
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [config["redirect_uri"]],
            }
        },
        scopes=SCOPES,
        redirect_uri=config["redirect_uri"],
    )


@auth_router.get("/auth/login")
def auth_login():
    try:
        flow = _build_flow()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    if flow.code_verifier:
        CODE_VERIFIER_STORE[state] = flow.code_verifier
    return RedirectResponse(auth_url)


@auth_router.get("/auth/callback")
def auth_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    flow = _build_flow()
    state = request.query_params.get("state")
    if state and state in CODE_VERIFIER_STORE:
        flow.code_verifier = CODE_VERIFIER_STORE.pop(state)
    flow.fetch_token(code=code)
    credentials = flow.credentials
    set_calendar_token(credentials.token, credentials.refresh_token)
    return RedirectResponse("/test/googleCalendar/index.html")


app = FastAPI(title="Google Calendar OAuth Test")
app.include_router(auth_router)
app.mount(
    "/test/googleCalendar",
    StaticFiles(directory=BASE_DIR, html=True),
    name="google-calendar-test-page",
)

import asyncio
import json
import queue
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel
from starlette.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from calendar_reminder import router as calendar_router
from question_service import handle_question
from test.googleCalendar.app import auth_router as google_calendar_auth_router


app = FastAPI(title="RAG Architecture PoC")
app.include_router(google_calendar_auth_router)
app.include_router(calendar_router)
app.mount(
    "/test/googleCalendar",
    StaticFiles(directory=Path(__file__).resolve().parent / "test" / "googleCalendar", html=True),
    name="google-calendar-demo",
)


class ChatRequest(BaseModel):
    query: str
    user_id: str = "USER-001"
    session_id: str = "default-session"


@app.post("/chat")
def chat(req: ChatRequest, stream: bool = Query(default=False)):
    """Run the same routed RAG workflow used by the Python service."""
    if stream:
        return _stream_chat(req)

    response = handle_question(req.query, current_user=req.user_id, session_id=req.session_id)
    result = response["result"]
    answer = result["answer"] if isinstance(result, dict) else result

    payload = {
        **response,
        "answer": answer,
    }
    # Return formatted JSON so the complete result payload is readable in curl.
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
    )


def _stream_chat(req: ChatRequest):
    """Stream progress and the final answer as plain text while the request runs."""
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def run_question() -> None:
        try:
            response = handle_question(
                req.query,
                current_user=req.user_id,
                session_id=req.session_id,
                progress_callback=lambda line: events.put(("progress", line)),
            )
            result = response.get("result", {})
            if isinstance(result, dict):
                output = result.get("answer", "")
            elif isinstance(result, str):
                output = result
            else:
                output = json.dumps(result, ensure_ascii=False)
            events.put(("answer", output))
        except Exception as exc:
            events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            events.put(("done", None))

    async def stream_output():
        worker = asyncio.create_task(asyncio.to_thread(run_question))
        while True:
            event_type, payload = await asyncio.to_thread(events.get)
            if event_type == "progress":
                yield f"{payload}\n"
            elif event_type == "answer":
                yield f"\n{payload}\n"
            elif event_type == "error":
                yield f"\nRequest failed: {payload}\n"
            elif event_type == "done":
                break

        await worker

    return StreamingResponse(stream_output(), media_type="text/plain; charset=utf-8")

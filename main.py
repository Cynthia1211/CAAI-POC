import asyncio
import queue

from fastapi import FastAPI, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from question_service import handle_question


app = FastAPI(title="RAG Architecture PoC")


class ChatRequest(BaseModel):
    query: str
    user_id: str = "USER-001"


@app.post("/chat")
def chat(req: ChatRequest, stream: bool = Query(default=False)):
    """Run the same routed RAG workflow used by the Python service."""
    if stream:
        return _stream_chat(req)

    response = handle_question(req.query, current_user=req.user_id)
    result = response["result"]
    answer = result["answer"] if isinstance(result, dict) else result

    return {
        **response,
        "answer": answer,
    }


def _stream_chat(req: ChatRequest):
    """Stream progress and the final answer as plain text while the request runs."""
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def run_question() -> None:
        try:
            handle_question(
                req.query,
                current_user=req.user_id,
                progress_callback=lambda line: events.put(("progress", line)),
            )
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
            elif event_type == "error":
                yield f"\nRequest failed: {payload}\n"
            elif event_type == "done":
                break

        await worker

    return StreamingResponse(stream_output(), media_type="text/plain; charset=utf-8")

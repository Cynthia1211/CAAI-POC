
import logging
from typing import Literal

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from polly_service import list_standard_voices, synthesize_speech


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice"])


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    engine: Literal["standard"] = "standard"
    voice_id: str = "Joanna"


@router.get("/options")
def get_voice_options():
    """Provide the voices and engines available for the demo."""
    try:
        return {
            "engines": ["standard"],
            "voices": list_standard_voices(),
        }
    except (BotoCoreError, ClientError):
        logger.exception("Unable to retrieve Polly voices")
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve voice options.",
        )


@router.post(
    "/synthesize",
    responses={200: {"content": {"audio/mpeg": {}}}},
)
def generate_speech(request: SynthesizeRequest):
    """Generate MP3 audio from the supplied text."""
    try:
        voices = list_standard_voices()

        if request.voice_id not in {voice["id"] for voice in voices}:
            raise HTTPException(
                status_code=422,
                detail="Voice is not available for Standard English (US).",
            )

        audio = synthesize_speech(
            text=request.text,
            engine=request.engine,
            voice_id=request.voice_id,
        )

        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-store"},
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (BotoCoreError, ClientError):
        logger.exception("Polly synthesis failed")
        raise HTTPException(
            status_code=502,
            detail="Speech generation failed.",
        )

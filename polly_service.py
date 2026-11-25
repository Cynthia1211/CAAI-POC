
import os

import boto3


def get_polly_client():
    """Create a Polly client using separate voice-service credentials."""
    session = boto3.Session(
        profile_name=os.getenv("POLLY_AWS_PROFILE") or None,
        region_name=os.getenv("POLLY_REGION", "us-east-2"),
    )
    return session.client("polly")


def list_standard_voices(language_code: str = "en-US") -> list[dict]:
    """Return available Standard voices for the selected language."""
    client = get_polly_client()
    paginator = client.get_paginator("describe_voices")

    voices = []

    for page in paginator.paginate(
        Engine="standard",
        LanguageCode=language_code,
    ):
        for voice in page["Voices"]:
            voices.append({
                "id": voice["Id"],
                "name": voice["Name"],
                "gender": voice.get("Gender"),
                "language_code": voice["LanguageCode"],
            })

    return sorted(voices, key=lambda voice: voice["name"])


def synthesize_speech(
    text: str,
    engine: str = "standard",
    voice_id: str = "Joanna",
) -> bytes:
    """Convert text to MP3 audio using Amazon Polly."""

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    if len(text) > 1000:
        raise ValueError("Text exceeds the demo limit.")

    if engine not in ("standard", "neural", "generative"):
        raise ValueError("Unsupported Polly engine.")

    client = get_polly_client()

    response = client.synthesize_speech(
        Text=text,
        Engine=engine,
        VoiceId=voice_id,
        OutputFormat="mp3",
    )

    audio_stream = response["AudioStream"]

    try:
        return audio_stream.read()
    finally:
        audio_stream.close()

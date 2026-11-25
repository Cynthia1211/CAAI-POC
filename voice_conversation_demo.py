
"""Multi-turn CAAI voice conversation demo."""

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from transcribe_service import main as transcribe_microphone


BASE_URL = "http://127.0.0.1:8000"
USER_ID = "USER-001"
VOICE_ID = "Joanna"
AUDIO_PATH = Path("/tmp/caai-voice-conversation.mp3")


def post_json(path: str, payload: dict, timeout: int = 120):
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=timeout) as response:
        return response.read()


def conversation_turn():
    """Record, transcribe, get an AI answer, and speak it."""

    # 1. Listen
    question = asyncio.run(transcribe_microphone())

    if not question:
        print("\nNo speech recognized. Try again.")
        return

    print(f"\nYou: {question}")

    # 2. Generate the AI response
    print("\nThinking...")

    raw_response = post_json(
        "/chat",
        {
            "query": question,
            "user_id": USER_ID,
        },
    )

    chat_data = json.loads(raw_response)

    answer = chat_data["answer"]
    answer = answer.split(
        "\n\nSources used:", 1
    )[0].strip()

    if not answer:
        print("The chatbot returned no answer.")
        return

    print(f"\nAssistant: {answer}")

    if len(answer) > 1000:
        print(
            "\nAnswer exceeds the current speech limit. "
            "Skipping audio for this turn."
        )
        return

    # 3. Convert AI response to speech
    print("\nSpeaking...")

    audio = post_json(
        "/voice/synthesize",
        {
            "text": answer,
            "engine": "standard",
            "voice_id": VOICE_ID,
        },
    )

    AUDIO_PATH.write_bytes(audio)

    # 4. Play audio
    subprocess.run(
        ["afplay", str(AUDIO_PATH)],
        check=True,
    )

    print("\nResponse completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically listen again after each answer",
    )
    args = parser.parse_args()

    print("\n=== CAAI Voice Conversation ===")
    print("Voice:", VOICE_ID)

    print(
        "\nThis demo sends speech to the team's AWS account "
        "and AI answers to your personal Polly AWS account."
    )

    approval = input(
        "Use approved synthetic demo data only. "
        "Type yes to continue: "
    )

    if approval.strip().lower() != "yes":
        print("Session cancelled.")
        return

    print("\nPress Ctrl+C anytime to exit.")

    turn = 0

    try:
        while True:
            if not args.auto:
                command = input(
                    "\nPress Enter to speak, "
                    "or type q to quit: "
                ).strip().lower()

                if command in ("q", "quit", "exit"):
                    break

            turn += 1
            print(f"\n--- Conversation Turn {turn} ---")

            try:
                conversation_turn()

            except (HTTPError, URLError) as exc:
                print(f"\nAPI error: {exc}")
                print("Check that FastAPI is running.")

            except Exception as exc:
                print(
                    f"\nTurn failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            if args.auto:
                print("\nListening again...")

    except KeyboardInterrupt:
        print("\nConversation interrupted.")

    print("\nConversation ended.")


if __name__ == "__main__":
    main()

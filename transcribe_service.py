
#Amazon Transcribe Streaming

import asyncio
import math
import os
from array import array

import sounddevice as sd

from smithy_http.aio.crt import AWSCRTHTTPClient
from aws_sdk_transcribe_streaming import (
    AsyncTranscribeStreamingClient,
    AsyncTranscribeStreamingConfig,
)
from aws_sdk_transcribe_streaming.models import (
    AudioEvent,
    AudioStreamAudioEvent,
    LanguageCode,
    MediaEncoding,
    TranscriptResultStreamTranscriptEvent,
)


AWS_REGION = "us-east-2"
MIC_DEVICE = 2

SAMPLE_RATE = int(
    sd.query_devices(MIC_DEVICE)["default_samplerate"]
)
BLOCK_SIZE = int(SAMPLE_RATE * 0.1)

# Automatic silence detection settings
SILENCE_SECONDS = 1.5
START_TIMEOUT_SECONDS = 8.0
MAX_RECORD_SECONDS = 25.0

# Adjustable microphone sensitivity
SPEECH_THRESHOLD = float(
    os.getenv("CAAI_VAD_THRESHOLD", "0.012")
)

# Avoid triggering on a single noisy audio block
MIN_SPEECH_BLOCKS = 2


def audio_level(chunk: bytes) -> float:
    """Calculate normalized RMS volume of 16-bit PCM audio."""
    samples = array("h")
    samples.frombytes(chunk)

    if not samples:
        return 0.0

    mean_square = sum(
        sample * sample for sample in samples
    ) / len(samples)

    return math.sqrt(mean_square) / 32768.0


async def send_microphone(stream):
    """Stream audio until speech ends or a timeout is reached."""
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        chunk = bytes(indata)
        loop.call_soon_threadsafe(
            audio_queue.put_nowait,
            chunk,
        )

    speech_started = False
    speech_blocks = 0
    last_speech_time = None

    try:
        with sd.RawInputStream(
            device=MIC_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            callback=callback,
        ):
            print("\nListening... Speak now!", flush=True)

            started_at = loop.time()

            while True:
                now = loop.time()
                elapsed = now - started_at

                if elapsed >= MAX_RECORD_SECONDS:
                    print("\nMaximum recording duration reached.")
                    break

                if (
                    not speech_started
                    and elapsed >= START_TIMEOUT_SECONDS
                ):
                    print("\nNo speech detected.")
                    break

                try:
                    chunk = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                level = audio_level(chunk)
                now = loop.time()

                if level >= SPEECH_THRESHOLD:
                    speech_blocks += 1

                    if speech_blocks >= MIN_SPEECH_BLOCKS:
                        if not speech_started:
                            speech_started = True
                            print(
                                "Speech detected...",
                                flush=True,
                            )

                        last_speech_time = now

                else:
                    speech_blocks = 0

                # Send audio to Amazon Transcribe
                await stream.input_stream.send(
                    AudioStreamAudioEvent(
                        value=AudioEvent(audio_chunk=chunk)
                    )
                )

                # Detect when user finishes speaking
                if (
                    speech_started
                    and last_speech_time is not None
                    and now - last_speech_time >= SILENCE_SECONDS
                ):
                    print(
                        "\nEnd of speech detected.",
                        flush=True,
                    )
                    break

    finally:
        # Closing input lets AWS finalize the transcript
        await stream.input_stream.close()


async def receive_transcripts(stream):
    """Collect finalized transcripts from Amazon Transcribe."""
    _, output_stream = await stream.await_output()

    if output_stream is None:
        raise RuntimeError("No transcript stream returned.")

    final_parts = []

    async for event in output_stream:
        if not isinstance(
            event, TranscriptResultStreamTranscriptEvent
        ):
            continue

        transcript = event.value.transcript

        if transcript is None:
            continue

        for result in transcript.results or []:
            if result.is_partial:
                continue

            alternatives = result.alternatives or []

            if alternatives and alternatives[0].transcript:
                recognized = (
                    alternatives[0].transcript.strip()
                )

                final_parts.append(recognized)

                print(
                    "Recognized:",
                    recognized,
                    flush=True,
                )

    return " ".join(final_parts).strip()


async def main():
    """Start one speech-to-text session."""
    sd.check_input_settings(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    print("\nAmazon Transcribe Streaming")
    print("Region:", AWS_REGION)
    print(
        "Microphone:",
        sd.query_devices(MIC_DEVICE)["name"],
    )
    print("Sample rate:", SAMPLE_RATE)
    print("Silence timeout:", SILENCE_SECONDS, "seconds")

    config = await AsyncTranscribeStreamingConfig.resolve(
        region=AWS_REGION,
        transport=AWSCRTHTTPClient(),
    )

    async with AsyncTranscribeStreamingClient(
        config=config
    ) as client:
        stream = await client.start_stream_transcription(
            language_code=LanguageCode.EN_US,
            media_sample_rate_hertz=SAMPLE_RATE,
            media_encoding=MediaEncoding.PCM,
        )

        async with stream:
            _, recognized_text = await asyncio.gather(
                send_microphone(stream),
                receive_transcripts(stream),
            )

    print("\nTranscription session finished.")

    return recognized_text


if __name__ == "__main__":
    result = asyncio.run(main())
    print("\nFinal text:", result)

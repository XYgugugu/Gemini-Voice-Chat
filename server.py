import os
import io
import sys
import wave
import asyncio
from typing import AsyncGenerator

import numpy as np
import librosa
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from google.genai.types import (
    LiveConnectConfig,
    HttpOptions,
    Modality,
    ContextWindowCompressionConfig,
    SlidingWindow,
)

# ---------------- Environment configuration & Gemini client ----------------

# Load environment variables (expects GEMINI_API_KEY or GOOGLE_API_KEY).
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment")

client = genai.Client(
    api_key=API_KEY,
    http_options=HttpOptions(api_version="v1alpha"),
)

MODEL = "models/gemini-2.0-flash-exp"
SYSTEM_PROMPT = "The conversation should focus on software development."

# All incoming audio will be normalized to mono, 16 kHz, 16-bit PCM.
TARGET_SAMPLE_RATE = 16000

# Enable context window compression so a single Live session can run longer.
CONFIG = LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction=SYSTEM_PROMPT,
    context_window_compression=ContextWindowCompressionConfig(
        sliding_window=SlidingWindow()
    ),
)

# ---------------- Timeouts ----------------
# Used only as protection when *no* response is received at all.
FIRST_CHUNK_TIMEOUT_SEC = 10.0

app = FastAPI()


# ---------------- Helpers ----------------
def extract_pcm_from_wav(
    wav_bytes: bytes, target_rate: int = TARGET_SAMPLE_RATE
) -> tuple[bytes, int]:
    """
    Convert a WAV file (as bytes) into mono, 16-bit PCM data at the target sample rate.

    The function:
    - Accepts mono or multi-channel input and multiple sample widths.
    - Converts audio to mono if necessary.
    - Resamples to the specified target_rate using librosa.resample.
    - Outputs 16-bit PCM (little-endian) bytes and the resulting sample rate.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_pcm = wf.readframes(n_frames)
    except wave.Error as e:
        raise ValueError(f"Invalid WAV file: {e}") from e

    # ---- Convert raw bytes to a float32 numpy array in [-1, 1] ----
    if sampwidth == 2:
        # 16-bit PCM
        audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        # 32-bit PCM (less common but supported)
        audio = np.frombuffer(raw_pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    # ---- Handle channels (downmix to mono if needed) ----
    if channels > 1:
        # Reshape to (num_samples, num_channels), then average across channels.
        audio = audio.reshape(-1, channels).mean(axis=1)

    # ---- Resample (if needed) using librosa ----
    if rate != target_rate:
        audio = librosa.resample(y=audio, orig_sr=rate, target_sr=target_rate)
        rate = target_rate

    # ---- Clip and convert back to int16 PCM bytes ----
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16).tobytes()

    return pcm16, rate


class LiveAudioSession:
    """
    Encapsulates a single long-lived Gemini Live API session for audio input/output.

    Multiple HTTP requests share this session, allowing the model to preserve
    conversational context across turns. A lock is used to ensure that only one
    /chat/audio call accesses the session at a time.
    """

    def __init__(self) -> None:
        # Async context manager returned by client.aio.live.connect.
        self._cm = None
        # The active live session instance.
        self.session = None
        # Ensures serialized access to the shared live session.
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Initialize the Live session if it does not already exist.
        """
        if self.session is not None:
            return
        self._cm = client.aio.live.connect(model=MODEL, config=CONFIG)
        self.session = await self._cm.__aenter__()
        print("[server] Live session started")

    async def stop(self) -> None:
        """
        Close the Live session if it is currently active.
        """
        if self.session is None or self._cm is None:
            return
        await self._cm.__aexit__(None, None, None)
        self.session = None
        self._cm = None
        print("[server] Live session stopped")

    async def stream_turn(
        self, pcm_bytes: bytes, rate: int
    ) -> AsyncGenerator[bytes, None]:
        """
        Send a single user audio turn into the Live session and yield the model's
        audio response as a stream of PCM chunks.

        There is no per-turn timeout here; the model is allowed to finish its
        response naturally. The HTTP handler applies a timeout only for the
        first response chunk.
        """
        if self.session is None:
            await self.start()

        mime_type = f"audio/pcm;rate={rate}"

        # Only one turn at a time is permitted through the live session.
        async with self.lock:
            # Send audio into the active Live session.
            await self.session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type=mime_type)
            )

            # Receive this turn's responses.
            # We rely on the SDK to end the async iterator when the turn is complete.
            turn = self.session.receive()

            async for msg in turn:
                # Newer SDKs: audio can be exposed as msg.data
                if getattr(msg, "data", None):
                    yield msg.data
                    continue

                # Fallback / future-proof: inspect server_content.model_turn.parts.
                server_content = getattr(msg, "server_content", None)
                if (
                    server_content
                    and getattr(server_content, "model_turn", None)
                    and getattr(server_content.model_turn, "parts", None)
                ):
                    for part in server_content.model_turn.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            yield inline.data


live_session = LiveAudioSession()


@app.on_event("startup")
async def _startup() -> None:
    """
    FastAPI startup hook: eagerly initialize the Live session when the server starts.
    """
    await live_session.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    """
    FastAPI shutdown hook: gracefully stop the Live session when the server stops.
    """
    await live_session.stop()


# ---------------- Main route ----------------
@app.post("/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    """
    Accept a WAV file upload, pass its raw PCM audio to the long-lived
    Gemini Live session, and stream the model's PCM audio response.

    If the first audio chunk from Gemini is not received within
    FIRST_CHUNK_TIMEOUT_SEC seconds, an HTTP 504 (Gateway Timeout) is returned.
    The Live session remains active so subsequent user audio can still be handled.
    """
    if file.content_type not in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/x-pn-wav",
        None,  # Some simple clients may omit this header.
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}",
        )

    wav_bytes = await file.read()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Decode WAV into raw PCM (mono, 16 kHz, 16-bit).
    try:
        pcm_bytes, rate = extract_pcm_from_wav(wav_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prepare a single-turn stream on the long-lived session.
    agen = live_session.stream_turn(pcm_bytes, rate)

    # ---- Timeout logic for the first chunk ----
    # We wait up to FIRST_CHUNK_TIMEOUT_SEC for the first audio chunk.
    # If it takes longer, we abort this HTTP request with 504 but
    # keep the Live session running for future requests.
    try:
        first_chunk = await asyncio.wait_for(
            agen.__anext__(), timeout=FIRST_CHUNK_TIMEOUT_SEC
        )
    except StopAsyncIteration:
        # No audio produced at all; treat as an empty response.
        await agen.aclose()
        return StreamingResponse(iter([]), media_type="audio/pcm")
    except asyncio.TimeoutError:
        # Timed out waiting for the first chunk: inform the client.
        await agen.aclose()
        raise HTTPException(
            status_code=504,
            detail=(
                f"Model audio response timed out after "
                f"{FIRST_CHUNK_TIMEOUT_SEC:.1f}s; session is still alive."
            ),
        )

    # At this point we have at least one chunk; now stream the remainder.
    async def full_stream() -> AsyncGenerator[bytes, None]:
        # Yield the first chunk that was already awaited.
        yield first_chunk
        # Then yield the remaining chunks until the generator finishes.
        async for chunk in agen:
            yield chunk

    # Note: we always return audio/pcm: raw 16-bit PCM at 24 kHz from Gemini.
    # The existing client.py plays this at 24 kHz.
    return StreamingResponse(full_stream(), media_type="audio/pcm")

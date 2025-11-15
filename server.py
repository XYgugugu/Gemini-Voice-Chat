# server.py
# Voice↔voice server with Gemini Live API (v1alpha)
# - Single long-lived Live session (same as original AudioLoop)
# - Same send_audio / receive_audio structure, preserving context.
# - Input: HTTP /chat/audio receives a WAV, converts to PCM16 16k, and feeds mic_queue.
# - Output: server STREAMS Gemini's 24 kHz PCM audio back to the client in real time.
#   (No server-side playback now.)

import asyncio
import os
import sys
import io
import traceback
import contextlib

import numpy as np
import soundfile as sf
import librosa

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig, HttpOptions, Modality

# ---------- Environment ----------
load_dotenv()  # expects GOOGLE_API_KEY in .env
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env")

# ---------- Python version check ----------
if sys.version_info < (3, 11, 0):
    print("Error: Python 3.11+ is required (uses asyncio.TaskGroup).")
    sys.exit(1)

# ---------- Tunables (copied from original) ----------
AUDIO_IN_QUEUE_SIZE = 256   # not used now, but kept for similarity
SPEAKER_FRAMES = 2048       # not used now

# ---------- Audio config ----------
SEND_SAMPLE_RATE = 16000     # "mic" input; each chunk labeled ';rate=16000'
RECEIVE_SAMPLE_RATE = 24000  # model output is 24 kHz PCM
MIC_FRAMES = 1024            # ~64 ms @ 16 kHz
BYTES_PER_SAMPLE = 2         # 16-bit PCM

# ---------- System prompt ----------
SYSTEM_PROMPT = "The conversation should focus on software development"

# ---------- API (v1alpha) ----------
client = genai.Client(
    http_options=HttpOptions(api_version="v1alpha")  # explicitly use v1alpha
)
MODEL = "models/gemini-2.0-flash-exp"

# Keep config minimal; v1alpha LiveConnectConfig may reject extra fields.
CONFIG = LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction=SYSTEM_PROMPT,
)

app = FastAPI(title="Gemini Voice Server (streaming audio to client)")


def wav_to_int16_mono_16k(wav_bytes: bytes) -> bytes:
    """
    Read arbitrary WAV, convert to mono 16 kHz PCM16 LE.
    Returns raw PCM16 bytes.
    """
    data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)  # [T, C]
    mono = data.mean(axis=1)  # average channels → mono

    if sr != SEND_SAMPLE_RATE:
        mono = librosa.resample(
            mono.astype(np.float32),
            orig_sr=sr,
            target_sr=SEND_SAMPLE_RATE,
        )

    mono = np.clip(mono, -1.0, 1.0)
    return (mono * 32767.0).astype(np.int16).tobytes()


class AudioLoop:
    """
    Very close to your original AudioLoop:

      - send_audio(): mic_queue → Gemini
      - receive_audio(): Gemini → per-turn queues

    Differences:
      - No listen_audio() from mic; instead, HTTP feeds mic_queue.
      - No play_audio() on server; instead, each turn's audio is streamed to the client.
    """

    def __init__(self):
        self.mic_queue: asyncio.Queue[bytes] | None = None
        self.session = None

        # For HTTP side: for each model turn, receive_audio will take the next queue
        # from turn_queues and push audio chunks into it.
        self.turn_queues: asyncio.Queue[asyncio.Queue[bytes] | None] = asyncio.Queue()

        # Optional lock to make sure turns don't overlap badly
        self.turn_lock = asyncio.Lock()

    async def send_audio(self):
        """Stream 'mic' bytes; server VAD handles turn ends when input stops."""
        mime = f"audio/pcm;rate={SEND_SAMPLE_RATE}"
        while True:
            pcm = await self.mic_queue.get()
            try:
                await self.session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type=mime)
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                traceback.print_exc()

    async def receive_audio(self):
        """Receive model responses and route them to the per-turn queue."""
        while True:
            turn = self.session.receive()
            try:
                # Get the queue for THIS model turn (provided by /chat/audio)
                turn_queue: asyncio.Queue[bytes] | None = None
                try:
                    turn_queue = await self.turn_queues.get()
                except asyncio.QueueEmpty:
                    turn_queue = None

                async for msg in turn:
                    if msg.data and turn_queue is not None:
                        # Push audio chunks to this turn's queue (client will stream them)
                        await turn_queue.put(msg.data)
                    if msg.text:
                        # Optional: print captions on server
                        print("Gemini:", msg.text, end="")
                print()

                # Signal end-of-turn to the client by pushing a sentinel (empty bytes)
                if turn_queue is not None:
                    await turn_queue.put(b"")

            except asyncio.CancelledError:
                raise
            except Exception:
                traceback.print_exc()

    async def feed_pcm_to_mic_queue(self, pcm16_16k: bytes):
        """Chunk PCM into MIC_FRAMES and feed into mic_queue."""
        chunk_size_bytes = MIC_FRAMES * BYTES_PER_SAMPLE
        for i in range(0, len(pcm16_16k), chunk_size_bytes):
            chunk = pcm16_16k[i:i + chunk_size_bytes]
            if not chunk:
                break
            await self.mic_queue.put(chunk)

    async def run(self):
        """Run a single long-lived Live session, like the original AudioLoop.run()."""
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.mic_queue = asyncio.Queue(maxsize=8)

                print("Voice chat server started (v1alpha).")
                print("HTTP clients can feed WAV turns; model context is preserved.")

                tg.create_task(self.send_audio())     # mic_queue → session
                tg.create_task(self.receive_audio())  # session → per-turn queues

                await asyncio.Future()  # run until interrupted

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            traceback.print_exception(eg)
        except Exception as e:
            print(f"An error occurred in AudioLoop.run(): {e}")
        finally:
            print("Voice chat session ended.")


audio_loop = AudioLoop()


@app.on_event("startup")
async def startup_event():
    # Start the Gemini Live pipeline in the background
    asyncio.create_task(audio_loop.run())


@app.post("/chat/audio")
async def chat_audio(file: UploadFile = File(..., description="WAV (any SR)")):
    """
    Client uploads a WAV file (one "user turn").
    - Server converts to mono 16 kHz PCM.
    - Feeds it into the existing Live session as a new "mic" turn.
    - STREAMS the model's response audio (24 kHz PCM16 mono) back to the client.

    Response: StreamingResponse with media_type "audio/pcm;rate=24000"
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        pcm16_16k = wav_to_int16_mono_16k(blob)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid WAV or conversion failed: {e!r}",
        )

    # Create a per-turn queue that receive_audio will fill with PCM chunks
    turn_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # We serialize turns with a lock to avoid overlapping turns on the same session
    await audio_loop.turn_lock.acquire()

    # Register this queue so the *next* model turn audio goes into it
    await audio_loop.turn_queues.put(turn_queue)

    # Feed user audio into mic_queue in the background (so streaming can start ASAP)
    asyncio.create_task(audio_loop.feed_pcm_to_mic_queue(pcm16_16k))

    async def audio_stream():
        try:
            while True:
                chunk = await turn_queue.get()
                # Empty bytes as sentinel: end-of-turn
                if chunk == b"":
                    break
                yield chunk
        finally:
            # Release the turn lock when we're done streaming this turn
            audio_loop.turn_lock.release()

    return StreamingResponse(
        audio_stream(),
        media_type=f"audio/pcm;rate={RECEIVE_SAMPLE_RATE}",
    )


@app.get("/health")
def health():
    return {"ok": True}

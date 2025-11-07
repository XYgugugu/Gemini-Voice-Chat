# Voice↔voice chat with Gemini Live API (v1alpha)
# - Single system prompt via LiveConnectConfig.system_instruction
# - Mic streaming via send_realtime_input(audio=..., mime_type="audio/pcm;rate=16000")
# - 24 kHz PCM playback
# - Lossless queue to avoid cuts
#
# pip install --upgrade google-genai==1.1.0 pyaudio python-dotenv

import asyncio
import os
import sys
import traceback
import contextlib
import pyaudio

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig, HttpOptions, Modality

# ---------- Environment ----------
load_dotenv()  # expects GOOGLE_API_KEY in .env

# ---------- Python version check ----------
if sys.version_info < (3, 11, 0):
    print("Error: Python 3.11+ is required (uses asyncio.TaskGroup).")
    sys.exit(1)

# ---------- Tunables ----------
AUDIO_IN_QUEUE_SIZE = 256   # increase if you want a deeper playback buffer
SPEAKER_FRAMES = 2048       # output buffer for smoother playback

# ---------- Audio config ----------
FORMAT = pyaudio.paInt16     # 16-bit PCM (little-endian)
CHANNELS = 1
SEND_SAMPLE_RATE = 16000     # mic input; each chunk labeled ';rate=16000'
RECEIVE_SAMPLE_RATE = 24000  # model output is 24 kHz PCM
MIC_FRAMES = 1024            # ~64 ms @ 16 kHz

# ---------- System prompt ----------
# adjust the prompt accordingly / use system_prompt.txt
SYSTEM_PROMPT = (
    "The conversation should focus on software development"
)

# ---------- API (v1alpha) ----------
client = genai.Client(
    http_options=HttpOptions(api_version="v1alpha")  # explicitly use v1alpha
)
MODEL = "models/gemini-2.0-flash-exp"

# Keep config minimal; v1alpha LiveConnectConfig may reject extra fields.
CONFIG = LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction=SYSTEM_PROMPT,  # simple string prompt is accepted
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self):
        self.audio_in_queue: asyncio.Queue[bytes] | None = None  # model → speaker
        self.mic_queue: asyncio.Queue[bytes] | None = None       # mic → model
        self.session = None
        self.mic_stream = None
        self.spk_stream = None

    async def listen_audio(self):
        """Continuously capture mic audio and queue raw bytes."""
        mic_info = pya.get_default_input_device_info()
        self.mic_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=MIC_FRAMES,
        )
        kwargs = {"exception_on_overflow": False}
        while True:
            data = await asyncio.to_thread(self.mic_stream.read, MIC_FRAMES, **kwargs)
            await self.mic_queue.put(data)

    async def send_audio(self):
        """Stream mic bytes; server VAD handles turn ends when you pause."""
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
        """Receive model responses and enqueue for playback (lossless back-pressure)."""
        while True:
            turn = self.session.receive()
            try:
                async for msg in turn:
                    if msg.data:
                        # Wait if buffer is full (no drops → no cutting)
                        await self.audio_in_queue.put(msg.data)
                    if msg.text:
                        # Some models include text captions; harmless to print
                        print("Gemini:", msg.text, end="")
                print()
            except asyncio.CancelledError:
                raise
            except Exception:
                traceback.print_exc()

    async def play_audio(self):
        """Play model audio as it arrives."""
        self.spk_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            frames_per_buffer=SPEAKER_FRAMES,
        )
        write = self.spk_stream.write
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue(maxsize=AUDIO_IN_QUEUE_SIZE)
                self.mic_queue = asyncio.Queue(maxsize=8)

                print("Voice chat started (v1alpha). Speak; pause to let it reply. Ctrl+C to quit.")
                print("Tip: use headphones to avoid feedback.")

                tg.create_task(self.listen_audio())   # mic → mic_queue
                tg.create_task(self.send_audio())     # mic_queue → session
                tg.create_task(self.receive_audio())  # session → audio_in_queue
                tg.create_task(self.play_audio())     # audio_in_queue → speaker

                await asyncio.Future()  # run until interrupted

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            if self.mic_stream:
                with contextlib.suppress(Exception):
                    self.mic_stream.close()
            traceback.print_exception(eg)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            with contextlib.suppress(Exception):
                if self.mic_stream:
                    self.mic_stream.stop_stream(); self.mic_stream.close()
            with contextlib.suppress(Exception):
                if self.spk_stream:
                    self.spk_stream.stop_stream(); self.spk_stream.close()
            print("Voice chat session ended.")


if __name__ == "__main__":
    try:
        main = AudioLoop()
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nChat terminated by user.")
    finally:
        pya.terminate()
        print("Audio resources released.")

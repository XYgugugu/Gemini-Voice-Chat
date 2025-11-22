import os
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import (
    LiveConnectConfig,
    HttpOptions,
    Modality,
    ContextWindowCompressionConfig,
    SlidingWindow,
)

# ---------------- Setup ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment")

client = genai.Client(
    api_key=API_KEY,
    http_options=HttpOptions(api_version="v1alpha"),
)

MODEL = "models/gemini-2.0-flash-exp"
SYSTEM_PROMPT = "The conversation should focus on software development. Keep responses concise."

CONFIG = LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction=SYSTEM_PROMPT,
    context_window_compression=ContextWindowCompressionConfig(
        sliding_window=SlidingWindow()
    ),
)

app = FastAPI()

class LiveAudioSession:
    def __init__(self):
        self.session = None
        self._cm = None
        self.audio_out_queue = asyncio.Queue() # Buffer for Gemini -> Client

    async def start(self):
        """Connects to Gemini Live."""
        self._cm = client.aio.live.connect(model=MODEL, config=CONFIG)
        self.session = await self._cm.__aenter__()

    async def stop(self):
        """Closes the Gemini Live session."""
        if self.session:
            await self._cm.__aexit__(None, None, None)
            self.session = None

    async def send_audio(self, pcm_data: bytes):
        """Send raw PCM audio to Gemini."""
        if self.session:
            await self.session.send_realtime_input(
                audio=types.Blob(data=pcm_data, mime_type="audio/pcm;rate=16000")
            )

    async def receive_loop(self):
        """Continuously pulls audio from Gemini and puts it in the queue."""
        try:
            while True:
                async for msg in self.session.receive():
                    # 1. Audio Data
                    if getattr(msg, "data", None):
                        await self.audio_out_queue.put(msg.data)
                        continue
                    
                    # 2. Server Content (Standard)
                    server_content = getattr(msg, "server_content", None)
                    if server_content:
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn:
                            for part in model_turn.parts:
                                inline = getattr(part, "inline_data", None)
                                if inline and inline.data:
                                    await self.audio_out_queue.put(inline.data)
                        
                        if getattr(server_content, "turn_complete", False):
                            # Optional: Signal end of turn if needed, 
                            # but for streaming audio we just keep going.
                            pass
        except Exception as e:
            print(f"[Gemini] Error in receive_loop: {e}")
            traceback.print_exc()

# ---------------- WebSocket Route ----------------
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[ws] Client connected")

    session = LiveAudioSession()
    await session.start()

    try:
        # --- Task 1: Client -> Gemini ---
        async def receive_from_client():
            try:
                while True:
                    # Wait for audio from Unity/Python Client
                    data = await websocket.receive_bytes()
                    # Send to Gemini (Non-blocking usually)
                    await session.send_audio(data)
            except WebSocketDisconnect:
                print("[ws] Client disconnected.")
                raise # Cancel other tasks
            except Exception as e:
                print(f"[ws] Error receiving from client: {e}")

        # --- Task 2: Gemini -> Queue ---
        # This runs inside the session object to keep logic encapsulated
        # We just await it here to keep it alive.
        async def run_gemini_receiver():
            await session.receive_loop()

        # --- Task 3: Queue -> Client ---
        async def send_to_client():
            try:
                while True:
                    # Wait for audio in the queue
                    audio_data = await session.audio_out_queue.get()
                    # Send to WebSocket
                    await websocket.send_bytes(audio_data)
            except Exception as e:
                print(f"[ws] Error sending to client: {e}")

        # Run all 3 tasks concurrently
        await asyncio.gather(receive_from_client(), run_gemini_receiver(), send_to_client())

    except WebSocketDisconnect:
        print("[ws] Session closed")
    finally:
        await session.stop()
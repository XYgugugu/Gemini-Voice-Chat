import asyncio
import websockets
import pyaudio
import sys
import queue

# ================= CONFIGURATION =================
SERVER_URL = "ws://127.0.0.1:8000/ws/chat"

# Audio Settings
INPUT_RATE = 16000 
OUTPUT_RATE = 24000 
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 512 # Smaller chunk size for lower latency

class ClientState:
    def __init__(self):
        self.is_recording = False

state = ClientState()

async def monitor_input():
    loop = asyncio.get_running_loop()
    print("   [Instruction] Press ENTER to toggle recording.")
    while True:
        await loop.run_in_executor(None, sys.stdin.readline)
        state.is_recording = not state.is_recording
        if state.is_recording:
            print("\n🔴 RECORDING... (Press Enter to stop)")
        else:
            print("\n⏸️  PAUSED (Listening...)")

async def run_client():
    # We use two separate PyAudio instances to avoid any internal locking
    p_in = pyaudio.PyAudio()
    p_out = pyaudio.PyAudio()

    # Queues for decoupled processing
    mic_queue = asyncio.Queue()
    speaker_queue = asyncio.Queue()

    print(f"Connecting to {SERVER_URL}...")

    try:
        # 1. Setup Mic
        input_stream = p_in.open(
            format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, 
            input=True, frames_per_buffer=CHUNK_SIZE
        )
        
        # 2. Setup Speaker
        output_stream = p_out.open(
            format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, 
            output=True, frames_per_buffer=CHUNK_SIZE
        )

        async with websockets.connect(SERVER_URL) as websocket:
            print("✅ CONNECTED!")
            asyncio.create_task(monitor_input())

            # --- Task 1: Mic -> Queue (Producer) ---
            async def mic_reader():
                loop = asyncio.get_running_loop()
                while True:
                    try:
                        # Read non-blocking from executor
                        data = await loop.run_in_executor(
                            None, 
                            lambda: input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        )
                        # Only put in queue if recording, otherwise drop
                        if state.is_recording:
                            mic_queue.put_nowait(data)
                        else:
                            # Drain queue if we just stopped recording to prevent stale audio
                            while not mic_queue.empty():
                                try: mic_queue.get_nowait()
                                except: pass
                    except Exception as e:
                        print(f"Mic Error: {e}")
                    
                    await asyncio.sleep(0.001) # Yield

            # --- Task 2: Queue -> WebSocket (Consumer) ---
            async def socket_sender():
                while True:
                    # Wait for data from mic_queue
                    data = await mic_queue.get()
                    await websocket.send(data)

            # --- Task 3: WebSocket -> Queue (Producer) ---
            async def socket_receiver():
                while True:
                    try:
                        data = await websocket.recv()
                        speaker_queue.put_nowait(data)
                    except Exception as e:
                        print(f"Socket Receive Error: {e}")
                        break

            # --- Task 4: Queue -> Speaker (Consumer) ---
            async def speaker_writer():
                loop = asyncio.get_running_loop()
                while True:
                    data = await speaker_queue.get()
                    if data:
                        await loop.run_in_executor(
                            None, output_stream.write, data
                        )

            # Run all 4 tasks concurrently
            await asyncio.gather(
                mic_reader(), 
                socket_sender(), 
                socket_receiver(), 
                speaker_writer()
            )

    except websockets.exceptions.ConnectionClosed:
        print("\n[DISCONNECTED] Server closed connection.")
    except KeyboardInterrupt:
        print("\n[STOPPING] User interrupted.")
    finally:
        # Cleanup
        try: input_stream.close(); p_in.terminate()
        except: pass
        try: output_stream.close(); p_out.terminate()
        except: pass

if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass
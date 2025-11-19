import os
import sys
import wave
import threading

import requests
import pyaudio

SERVER_URL = "http://127.0.0.1:8000/chat/audio"
WAV_EXT = ".wav"

# Playback parameters (must match the server's audio streaming format)
PLAY_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16  # 16-bit PCM samples
CHUNK_BYTES = 4096        # Size of each HTTP chunk read from the stream

# Shared PyAudio instance for the entire client
pya = pyaudio.PyAudio()

# -------- Recording configuration (used when input line is empty) --------
REC_SAMPLE_RATE = 16000
REC_CHANNELS = 1
REC_FORMAT = pyaudio.paInt16
REC_FRAMES = 1024


def record_until_enter(outfile: str = "recorded.wav") -> str | None:
    """
    Record audio from the default microphone until the user presses ENTER again.

    The recording starts immediately and stops when the user presses ENTER
    a second time. The audio is saved as a WAV file at `outfile`, and the
    function returns the absolute path to that file. If no audio was captured,
    it returns None.
    """
    print("\n[Recording] Press ENTER again to stop...")

    frames = []
    recording = True

    def wait_for_enter():
        """Block until the user presses ENTER (or interrupts), then stop recording."""
        nonlocal recording
        try:
            input()  # Wait for the second ENTER to stop recording
        except (EOFError, KeyboardInterrupt):
            pass
        recording = False

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    stream = pya.open(
        format=REC_FORMAT,
        channels=REC_CHANNELS,
        rate=REC_SAMPLE_RATE,
        input=True,
        frames_per_buffer=REC_FRAMES,
    )

    try:
        while recording:
            data = stream.read(REC_FRAMES, exception_on_overflow=False)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()

    if not frames:
        print("[!] No audio recorded.")
        return None

    with wave.open(outfile, "wb") as wf:
        wf.setnchannels(REC_CHANNELS)
        wf.setsampwidth(pya.get_sample_size(REC_FORMAT))
        wf.setframerate(REC_SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    abs_path = os.path.abspath(outfile)
    print(f"[+] Recorded audio saved to {abs_path}\n")
    return abs_path


def send_wav_and_stream_play(file_path: str):
    """
    Send a WAV file to the server and play the streamed PCM response in real time.
    """
    if not os.path.isfile(file_path):
        print(f"[!] File not found: {file_path}")
        return

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/wav")}
            print(f"[*] Sending {file_path} to {SERVER_URL} ...")
            resp = requests.post(SERVER_URL, files=files, stream=True, timeout=None)
    except requests.RequestException as e:
        print(f"[!] Request error: {e}")
        return

    if resp.status_code != 200:
        print(f"[!] Server returned {resp.status_code}: {resp.text}")
        return

    ctype = resp.headers.get("Content-Type", "")
    print(f"[*] Server response Content-Type: {ctype}")

    # Open an output stream to play the server's PCM audio response
    stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=PLAY_SAMPLE_RATE,
        output=True,
    )

    try:
        for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                # Empty chunk indicates the server has finished streaming
                break
            stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        resp.close()

    print("[*] Playback finished.\n")


def interactive_loop():
    """
    Run the interactive CLI loop for sending WAV files or recording new audio.
    """
    print("Voice chat client (streaming playback)")
    print("Type a base name like 't1' to send 't1.wav' to the server.")
    print("Press ENTER with no text to RECORD, then ENTER again to stop.")
    print("Type 'q' or 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("Enter WAV base name (or ENTER to record)> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[+] Exiting client.")
            break

        # Handle exit commands
        if user_input.lower() in ("q", "quit", "exit"):
            print("[+] Bye.")
            break

        # If the user presses ENTER with no text: record from the mic, then send recorded.wav
        if user_input == "":
            recorded_path = record_until_enter("recorded.wav")
            if recorded_path is not None:
                send_wav_and_stream_play(recorded_path)
            continue

        # Otherwise, treat the input as the base name (or filename) of a WAV file
        if user_input.lower().endswith(WAV_EXT):
            wav_name = user_input
        else:
            wav_name = user_input + WAV_EXT

        file_path = os.path.abspath(wav_name)
        send_wav_and_stream_play(file_path)


def main():
    try:
        # If a filename is provided on the command line, run once and exit.
        # Otherwise, start the interactive loop.
        if len(sys.argv) > 1:
            file_path = os.path.abspath(sys.argv[1])
            send_wav_and_stream_play(file_path)
        else:
            interactive_loop()
    finally:
        pya.terminate()
        print("Audio resources released.")


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("Python 3.8+ is recommended.")
    main()

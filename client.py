#!/usr/bin/env python3
# client.py
import os
import sys

import requests
import pyaudio

SERVER_URL = "http://127.0.0.1:8000/chat/audio"
WAV_EXT = ".wav"

# Must match server's streaming format
PLAY_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHUNK_BYTES = 4096        # HTTP chunk size for streaming

pya = pyaudio.PyAudio()


def send_wav_and_stream_play(file_path: str):
    """Send WAV to server and play streaming PCM response in real time."""
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

    # Setup audio output
    stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=PLAY_SAMPLE_RATE,
        output=True,
    )

    try:
        for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                # Stream ended
                break
            stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        resp.close()

    print("[*] Playback finished.\n")


def interactive_loop():
    print("Voice chat client (streaming playback)")
    print("Type a base name like 't1' to send 't1.wav' to the server.")
    print("Type 'q' or 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("Enter WAV base name> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[+] Exiting client.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("[+] Bye.")
            break

        if user_input.lower().endswith(WAV_EXT):
            wav_name = user_input
        else:
            wav_name = user_input + WAV_EXT

        file_path = os.path.abspath(wav_name)
        send_wav_and_stream_play(file_path)


def main():
    try:
        # Single-shot mode if a filename is provided
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

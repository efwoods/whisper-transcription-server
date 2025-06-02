# realtime_whisper_server.py

import asyncio
import websockets
import numpy as np
import tempfile
import wave
import torch
import whisper
import threading
import soundfile as sf
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION * 2  # 16-bit (2 bytes) mono audio

model = whisper.load_model("base", device=device)
clients = set()

def transcribe_and_send(audio_data, websocket, loop):
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with wave.open(tmpfile.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)

    # Check audio duration and amplitude
    try:
        data, samplerate = sf.read(tmpfile.name)
        duration = len(data) / samplerate
        amplitude = float(np.max(np.abs(data)))
        print(f"Audio duration: {duration:.2f} sec")
        print(f"Max amplitude: {amplitude}")
    except Exception as e:
        print(f"[!] Error loading audio: {e}")
        return

    print(f"[+] Transcribing {tmpfile.name}")
    result = model.transcribe(tmpfile.name, language="en", fp16=torch.cuda.is_available())
    transcript = result.get("text", "").strip()

    # Add Whisper diagnostic info
    segments = result.get("segments", [])
    confidence = segments[0].get("no_speech_prob", 0.0) if segments else 0.0
    avg_logprob = segments[0].get("avg_logprob", 0.0) if segments else 0.0

    print(f"[+] Transcript: {transcript}")
    print(f"    ↳ no_speech_prob: {confidence}, avg_logprob: {avg_logprob}")

    result_json = {
        "transcript": transcript,
        "duration": duration,
        "amplitude": amplitude,
        "no_speech_prob": confidence,
        "avg_logprob": avg_logprob,
    }

    asyncio.run_coroutine_threadsafe(
        websocket.send(json.dumps(result_json)),
        loop
    )
async def handle_connection(websocket):
    print("[+] Client connected")
    clients.add(websocket)

    # Get the main event loop
    loop = asyncio.get_event_loop()

    try:
        buffer = bytearray()
        async for message in websocket:
            buffer.extend(message)

            while len(buffer) >= CHUNK_SIZE:
                chunk = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]

                # Pass the loop explicitly to avoid RuntimeError
                threading.Thread(target=transcribe_and_send, args=(chunk, websocket, loop)).start()

    except websockets.ConnectionClosed:
        print("[-] Client disconnected")
    finally:
        clients.remove(websocket)

async def main():
    print("[*] Server started on ws://0.0.0.0:8765")
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())

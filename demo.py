#!/usr/bin/env python3
"""
OVMS Workshop Demo
  Whisper (OVMS) -> TinyLlama (OVMS)

Usage:
  python demo.py            # microphone input mode
  python demo.py audio.wav  # WAV file mode
  python demo.py --text     # text-only mode
"""
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from openai import OpenAI

OVMS_URLS = ("http://127.0.0.1:8000/v3", "http://localhost:8000/v3")
LLM_MODEL = "tinyllama"
WHISPER_MODEL = "whisper-base-fp16-ov"
SAMPLE_RATE = 16000
MODEL_WAIT_TIMEOUT = 60
MODEL_WAIT_INTERVAL = 2


def record_audio(seconds: int = 5) -> np.ndarray:
    import sounddevice as sd

    print(f"Recording ({seconds}s)...", end=" ", flush=True)
    data = sd.rec(
        SAMPLE_RATE * seconds,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("done")
    return data.flatten()


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def connect_chat_client() -> OpenAI:
    last_error = None
    for url in OVMS_URLS:
        try:
            client = OpenAI(base_url=url, api_key="none")
            deadline = time.time() + MODEL_WAIT_TIMEOUT
            while True:
                models = client.models.list()
                names = [model.id for model in models.data]
                print(f"OVMS connected ({url}): {names}")
                if LLM_MODEL in names and WHISPER_MODEL in names:
                    return client
                if time.time() >= deadline:
                    print("OVMS is reachable but required models are not fully loaded.")
                    print(f"  -> expected: {LLM_MODEL}, {WHISPER_MODEL}")
                    print("  -> Run `python setup_ovms.py` and `wsl docker compose restart ovms`.")
                    sys.exit(1)
                print("  -> waiting for OVMS to finish loading models...")
                time.sleep(MODEL_WAIT_INTERVAL)
        except Exception as exc:
            last_error = exc

    print(f"OVMS connection failed: {last_error}")
    print("  -> Run `wsl docker compose up -d ovms` and try again.")
    sys.exit(1)


def transcribe_audio(client: OpenAI, audio_path: str | Path) -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
        )
    return transcript.text.strip()


def main() -> None:
    text_mode = "--text" in sys.argv
    wav_file = next((arg for arg in sys.argv[1:] if arg.lower().endswith(".wav")), None)

    client = connect_chat_client()
    print("Ready (Ctrl+C to stop)\n")

    while True:
        temp_wav = None
        text = ""

        try:
            if wav_file:
                audio_path = wav_file
            elif text_mode:
                try:
                    text = input("You: ").strip()
                except EOFError:
                    print()
                    break
                audio_path = None
            else:
                input("Press Enter to record...")
                audio = record_audio()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_wav = temp_file.name
                write_wav(temp_wav, audio)
                audio_path = temp_wav

            if audio_path is not None and not text_mode:
                text = transcribe_audio(client, audio_path)
                print(f"[Whisper] {text}\n")

            if not text:
                if wav_file:
                    print("[Whisper] Empty transcription.")
                    break
                continue

            print("[TinyLlama] ", end="", flush=True)
            for chunk in client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": text}],
                max_tokens=256,
                stream=True,
            ):
                print(chunk.choices[0].delta.content or "", end="", flush=True)
            print("\n")

            if wav_file:
                break
        finally:
            if temp_wav is not None:
                Path(temp_wav).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

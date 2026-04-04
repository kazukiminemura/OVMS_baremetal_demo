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
import wave
from pathlib import Path

import numpy as np
from openai import OpenAI

LLM_OVMS_URLS = ("http://127.0.0.1:8000/v3", "http://localhost:8000/v3")
WHISPER_OVMS_URLS = ("http://127.0.0.1:8001/v3", "http://localhost:8001/v3")
LLM_MODEL = "tinyllama"
WHISPER_MODEL = "OpenVINO/whisper-base-fp16-ov"
SAMPLE_RATE = 16000


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
    for url in LLM_OVMS_URLS:
        try:
            client = OpenAI(base_url=url, api_key="none")
            models = client.models.list()
            names = [model.id for model in models.data]
            print(f"OVMS chat connected ({url}): {names}")
            if LLM_MODEL not in names:
                print(f"OVMS chat is reachable but model '{LLM_MODEL}' is not loaded.")
                print("  -> Run `docker compose restart ovms` and try again.")
                sys.exit(1)
            return client
        except Exception as exc:
            last_error = exc

    print(f"OVMS chat connection failed: {last_error}")
    print("  -> Run `docker compose up -d ovms` and try again.")
    sys.exit(1)


def transcribe_audio(audio_path: str | Path) -> str:
    last_error = None
    for url in WHISPER_OVMS_URLS:
        try:
            candidate = OpenAI(base_url=url, api_key="none")
            with open(audio_path, "rb") as audio_file:
                transcript = candidate.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                )
            print(f"OVMS speech connected ({url})")
            return transcript.text.strip()
        except Exception as exc:
            last_error = exc

    print(f"OVMS speech connection failed: {last_error}")
    print("  -> Run `docker compose up -d ovms-whisper` and try again.")
    sys.exit(1)


def main() -> None:
    text_mode = "--text" in sys.argv
    wav_file = next((arg for arg in sys.argv[1:] if arg.lower().endswith(".wav")), None)

    chat_client = connect_chat_client()
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
                text = transcribe_audio(audio_path)
                print(f"[Whisper] {text}\n")

            if not text:
                if wav_file:
                    print("[Whisper] Empty transcription.")
                    break
                continue

            print("[TinyLlama] ", end="", flush=True)
            for chunk in chat_client.chat.completions.create(
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

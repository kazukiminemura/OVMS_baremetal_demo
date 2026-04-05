#!/usr/bin/env python3
"""
OVMS Workshop Demo
  Whisper (OVMS) -> selected LLM (OVMS)

Usage:
  python demo.py            # microphone input mode
  python demo.py audio.wav  # WAV file mode
  python demo.py --text     # text-only mode
  python demo.py --model qwen3-4b
"""
import sys
import os
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import httpx
from openai import OpenAI
from openai import APIConnectionError

OVMS_URLS = ("http://127.0.0.1:8000/v3", "http://localhost:8000/v3")
SUPPORTED_LLM_MODELS = {
    "tinyllama",
    "qwen3-4b",
    "llama-3.2-3b-instruct",
}
DEFAULT_LLM_MODEL = os.environ.get("OVMS_LLM_MODEL", "tinyllama")
WHISPER_MODEL = "whisper-base-fp16-ov"
SAMPLE_RATE = 16000
MODEL_WAIT_TIMEOUT = 60
MODEL_WAIT_INTERVAL = 2
CHAT_RETRY_COUNT = 2
TRANSCRIBE_RETRY_COUNT = 3


def parse_llm_model(argv: list[str]) -> str:
    if "--model" in argv:
        model_index = argv.index("--model")
        if model_index + 1 >= len(argv):
            raise SystemExit("Error: --model requires a value.")
        model_name = argv[model_index + 1]
    else:
        model_name = DEFAULT_LLM_MODEL

    if model_name not in SUPPORTED_LLM_MODELS:
        supported = ", ".join(sorted(SUPPORTED_LLM_MODELS))
        raise SystemExit(f"Error: unsupported LLM model '{model_name}'. Choose from: {supported}")

    return model_name


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


def connect_chat_client(llm_model: str, exit_on_failure: bool = True) -> OpenAI:
    last_error = None
    for url in OVMS_URLS:
        try:
            client = OpenAI(base_url=url, api_key="none")
            deadline = time.time() + MODEL_WAIT_TIMEOUT
            while True:
                models = client.models.list()
                names = [model.id for model in models.data]
                print(f"OVMS connected ({url}): {names}")
                if llm_model in names and WHISPER_MODEL in names:
                    return client
                if time.time() >= deadline:
                    if exit_on_failure:
                        print("OVMS is reachable but required models are not fully loaded.")
                        print(f"  -> expected: {llm_model}, {WHISPER_MODEL}")
                        print("  -> Run `python setup_ovms.py`, then restart OVMS with `./start_ovms.ps1`.")
                        sys.exit(1)
                    raise RuntimeError("OVMS models are not fully loaded yet.")
                print("  -> waiting for OVMS to finish loading models...")
                time.sleep(MODEL_WAIT_INTERVAL)
        except Exception as exc:
            last_error = exc

    if exit_on_failure:
        print(f"OVMS connection failed: {last_error}")
        print("  -> Start OVMS with `./start_ovms.ps1` and try again.")
        sys.exit(1)
    raise RuntimeError(f"OVMS connection failed: {last_error}")


def transcribe_audio(client: OpenAI, llm_model: str, audio_path: str | Path) -> tuple[str, OpenAI]:
    for attempt in range(TRANSCRIBE_RETRY_COUNT + 1):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                )
            return transcript.text.strip(), client
        except (APIConnectionError, httpx.HTTPError):
            if attempt >= TRANSCRIBE_RETRY_COUNT:
                raise
            print("[system] OVMS connection lost during transcription. Reconnecting...")
            time.sleep(MODEL_WAIT_INTERVAL)
            client = connect_chat_client(llm_model, exit_on_failure=False)
        except Exception as exc:
            message = str(exc)
            if "Mediapipe graph definition with requested name is not found" in message:
                raise RuntimeError(
                    "OVMS speech-to-text is not deployed for the requested Whisper model. "
                    "Run `python setup_ovms.py`, restart OVMS with `./start_ovms.ps1`, and try again."
                ) from exc
            raise
    return "", client


def stream_chat_completion(client: OpenAI, llm_model: str, text: str):
    return client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": text}],
        max_tokens=256,
        stream=True,
    )


def create_chat_completion(client: OpenAI, llm_model: str, text: str):
    return client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": text}],
        max_tokens=256,
        stream=False,
    )


def reconnect_client(llm_model: str) -> OpenAI:
    print("[system] OVMS connection lost. Reconnecting...\n")
    time.sleep(MODEL_WAIT_INTERVAL)
    return connect_chat_client(llm_model, exit_on_failure=False)


def main() -> None:
    llm_model = parse_llm_model(sys.argv[1:])
    text_mode = "--text" in sys.argv
    stream_mode = "--stream" in sys.argv
    wav_file = next((arg for arg in sys.argv[1:] if arg.lower().endswith(".wav")), None)

    client = connect_chat_client(llm_model)
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
                text, client = transcribe_audio(client, llm_model, audio_path)
                print(f"[Whisper] {text}\n")

            if not text:
                if wav_file:
                    print("[Whisper] Empty transcription.")
                    break
                continue

            print(f"[{llm_model}] ", end="", flush=True)
            for attempt in range(CHAT_RETRY_COUNT + 1):
                try:
                    if stream_mode:
                        for chunk in stream_chat_completion(client, llm_model, text):
                            print(chunk.choices[0].delta.content or "", end="", flush=True)
                    else:
                        response = create_chat_completion(client, llm_model, text)
                        print(response.choices[0].message.content or "", end="", flush=True)
                    print("\n")
                    break
                except (APIConnectionError, httpx.RemoteProtocolError):
                    if attempt >= CHAT_RETRY_COUNT:
                        if stream_mode:
                            print("\n[system] Streaming failed. Falling back to non-streaming response...\n")
                            response = create_chat_completion(client, llm_model, text)
                            print(response.choices[0].message.content or "", end="", flush=True)
                            print("\n")
                            break
                        raise
                    client = reconnect_client(llm_model)
                    print(f"[{llm_model}] ", end="", flush=True)
                except httpx.HTTPError:
                    if attempt >= CHAT_RETRY_COUNT:
                        raise
                    client = reconnect_client(llm_model)
                    print(f"[{llm_model}] ", end="", flush=True)
                except RuntimeError as exc:
                    print(f"\n[system] {exc}")
                    print("  -> Start OVMS with `./start_ovms.ps1` and try again.\n")
                    break

            if wav_file:
                break
        except (APIConnectionError, RuntimeError):
            print("\n[system] Inference failed.")
            print("  -> Start OVMS with `./start_ovms.ps1` and try again.\n")
            if wav_file:
                break
        finally:
            if temp_wav is not None:
                Path(temp_wav).unlink(missing_ok=True)

if __name__ == "__main__":
    main()

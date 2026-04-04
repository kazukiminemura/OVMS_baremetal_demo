#!/usr/bin/env python3
"""
OVMS Workshop Demo
  Whisper base (OpenVINO GenAI) → TinyLlama (OVMS)

使い方:
  python demo.py            # マイク入力モード
  python demo.py audio.wav  # WAVファイル入力モード
  python demo.py --text     # テキスト入力モード（マイクなし）
"""
import sys
import wave
import numpy as np
import openvino_genai as ov
from openai import OpenAI

OVMS_URL    = "http://localhost:8000/v3"
SAMPLE_RATE = 16000


def record_audio(seconds: int = 5) -> np.ndarray:
    import sounddevice as sd
    print(f"録音中 ({seconds}秒)...", end=" ", flush=True)
    data = sd.rec(SAMPLE_RATE * seconds, samplerate=SAMPLE_RATE,
                  channels=1, dtype="float32")
    sd.wait()
    print("完了")
    return data.flatten()


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as f:
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def main():
    text_mode = "--text" in sys.argv
    wav_file  = next((a for a in sys.argv[1:] if a.endswith(".wav")), None)

    print("Whisper を読み込み中...")
    whisper = ov.WhisperPipeline("models/whisper", "CPU")
    llm     = OpenAI(base_url=OVMS_URL, api_key="none")

    print("準備完了! (Ctrl+C で終了)\n")

    while True:
        # --- 音声 or テキスト入力 ---
        if wav_file:
            audio = load_wav(wav_file)
        elif text_mode:
            text = input("あなた: ").strip()
            audio = None
        else:
            input("Enterキーで話す...")
            audio = record_audio()

        # --- Whisper 文字起こし ---
        if audio is not None:
            text = whisper.generate(audio).strip()
            print(f"[Whisper] {text}\n")

        if not text:
            continue

        # --- TinyLlama 応答 (OVMS) ---
        print("[TinyLlama] ", end="", flush=True)
        for chunk in llm.chat.completions.create(
            model="tinyllama",
            messages=[{"role": "user", "content": text}],
            max_tokens=256,
            stream=True,
        ):
            print(chunk.choices[0].delta.content or "", end="", flush=True)
        print("\n")

        if wav_file:
            break  # ファイルモードは1回で終了


if __name__ == "__main__":
    main()

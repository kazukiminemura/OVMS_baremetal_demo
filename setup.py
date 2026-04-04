#!/usr/bin/env python3
"""
モデルのダウンロード＆OpenVINO変換スクリプト
実行: python setup.py

必要: pip install 'optimum[openvino]>=1.18'
"""
import subprocess, sys, os, shutil

MODELS = [
    {
        "name": "TinyLlama (LLM)",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "out": "models/tinyllama",
        "args": ["--weight-format", "int8", "--task", "text-generation-with-past"],
        "required_files": [
            "openvino_model.xml",
            "openvino_model.bin",
            "openvino_tokenizer.xml",
            "openvino_tokenizer.bin",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
        ],
    },
    {
        "name": "Whisper base (音声認識)",
        "model_id": "openai/whisper-base",
        "out": "models/whisper",
        "args": ["--task", "automatic-speech-recognition"],
        "required_files": [
            "openvino_encoder_model.xml",
            "openvino_encoder_model.bin",
            "openvino_decoder_model.xml",
            "openvino_decoder_model.bin",
            "openvino_tokenizer.xml",
            "openvino_tokenizer.bin",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
        ],
    },
]

cli = shutil.which("optimum-cli")
if cli is None:
    print("エラー: optimum-cli が見つかりません")
    print("  pip install 'optimum[openvino]>=1.18' を実行してください")
    sys.exit(1)

for m in MODELS:
    if os.path.isdir(m["out"]) and all(
        os.path.exists(os.path.join(m["out"], required))
        for required in m["required_files"]
    ):
        print(f"[スキップ] {m['name']} は変換済み")
        continue
    print(f"[変換中] {m['name']} ...")
    cmd = [cli, "export", "openvino", "-m", m["model_id"]] + m["args"] + [m["out"]]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"エラー: {m['name']} の変換に失敗しました")
        sys.exit(1)
    print(f"[完了] {m['out']}")

print("\n全モデルの準備が完了しました!")
print("次のステップ: docker compose up -d")

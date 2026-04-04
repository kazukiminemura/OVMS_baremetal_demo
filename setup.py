#!/usr/bin/env python3
"""
TinyLlama export script for OVMS demo.

Required:
  pip install "optimum[openvino]>=1.18"
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = {
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
}


def has_required_files(model_dir: str, required_files: list[str]) -> bool:
    return os.path.isdir(model_dir) and all(
        os.path.exists(os.path.join(model_dir, required))
        for required in required_files
    )


scripts_dir = Path(sys.executable).resolve().parent
cli = (
    shutil.which("optimum-cli")
    or shutil.which("optimum-cli.exe")
    or next(
        (
            str(path)
            for path in (scripts_dir / "optimum-cli.exe", scripts_dir / "optimum-cli")
            if path.exists()
        ),
        None,
    )
)

if cli is None:
    print("Error: optimum-cli was not found")
    print('  Run `pip install "optimum[openvino]>=1.18"` and try again.')
    sys.exit(1)

if has_required_files(MODEL["out"], MODEL["required_files"]):
    print(f"[skip] {MODEL['name']} is already exported")
    sys.exit(0)

print(f"[exporting] {MODEL['name']} ...")
cmd = [cli, "export", "openvino", "-m", MODEL["model_id"]] + MODEL["args"] + [MODEL["out"]]
env = os.environ.copy()
env["PYTHONUTF8"] = "1"
result = subprocess.run(cmd, env=env)
if result.returncode != 0:
    print(f"Error: failed to export {MODEL['name']}")
    sys.exit(1)

print(f"[done] {MODEL['out']}")
print("Next: docker compose --profile serve up -d ovms ovms-whisper")

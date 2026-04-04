#!/usr/bin/env python3
"""
Prepare a single OVMS model repository with TinyLlama and Whisper.

Requirements:
  pip install "optimum[openvino]>=1.18"
  docker
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_REPOSITORY = Path("models")
TINYLLAMA_DIR = MODEL_REPOSITORY / "tinyllama"
CONFIG_PATH = MODEL_REPOSITORY / "config.json"
WHISPER_SOURCE_MODEL = "OpenVINO/whisper-base-fp16-ov"
WHISPER_MODEL_NAME = "whisper-base-fp16-ov"
OVMS_IMAGE = "openvino/model_server:latest-gpu"
TARGET_DEVICE = os.environ.get("OVMS_TARGET_DEVICE", "GPU").upper()
DOCKER_CMD = shlex.split(os.environ.get("OVMS_DOCKER_CMD", "wsl docker"))

TINYLLAMA_REQUIRED_FILES = [
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
]

TINYLLAMA_GRAPH = """input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
    [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {
      models_path: "/models/tinyllama",
      cache_size: 10,
      max_num_batched_tokens: 512,
      max_num_seqs: 256,
      device: "__TARGET_DEVICE__"
    }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler"
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}
"""
TINYLLAMA_GRAPH = TINYLLAMA_GRAPH.replace("__TARGET_DEVICE__", TARGET_DEVICE)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PYTHONUTF8": "1"})
    if result.returncode != 0:
        sys.exit(result.returncode)


def capture(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result.stdout.strip()


def find_optimum_cli() -> str:
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
        print('Error: optimum-cli was not found. Run `pip install "optimum[openvino]>=1.18"`.')
        sys.exit(1)
    return cli


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        print(f"Error: `{name}` was not found in PATH.")
        sys.exit(1)


def validate_target_device() -> None:
    if TARGET_DEVICE not in {"CPU", "GPU"}:
        print("Error: OVMS_TARGET_DEVICE must be CPU or GPU.")
        sys.exit(1)


def get_repo_mount_path() -> str:
    repo_path = MODEL_REPOSITORY.resolve()
    if DOCKER_CMD[:1] == ["wsl"]:
        return capture(["wsl", "wslpath", "-a", str(repo_path)])
    return str(repo_path)


def has_required_files(model_dir: Path, required_files: list[str]) -> bool:
    return model_dir.is_dir() and all((model_dir / required).exists() for required in required_files)


def export_tinyllama() -> None:
    if has_required_files(TINYLLAMA_DIR, TINYLLAMA_REQUIRED_FILES):
        print("[skip] TinyLlama is already exported")
    else:
        MODEL_REPOSITORY.mkdir(parents=True, exist_ok=True)
        cli = find_optimum_cli()
        run(
            [
                cli,
                "export",
                "openvino",
                "-m",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--weight-format",
                "int8",
                "--task",
                "text-generation-with-past",
                str(TINYLLAMA_DIR),
            ]
        )

    (TINYLLAMA_DIR / "graph.pbtxt").write_text(TINYLLAMA_GRAPH, encoding="utf-8")


def write_base_config() -> None:
    MODEL_REPOSITORY.mkdir(parents=True, exist_ok=True)
    config = {
        "model_config_list": [
            {
                "config": {
                    "name": "tinyllama",
                    "base_path": "/models/tinyllama",
                }
            },
            {
                "config": {
                    "name": WHISPER_MODEL_NAME,
                    "base_path": f"/models/OpenVINO/{WHISPER_MODEL_NAME}",
                }
            }
        ]
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")


def prepare_whisper() -> None:
    repo_mount = f"{get_repo_mount_path()}:/models"
    run(
        DOCKER_CMD
        + [
            "run",
            "--rm",
            "-v",
            repo_mount,
            OVMS_IMAGE,
            "--pull",
            "--source_model",
            WHISPER_SOURCE_MODEL,
            "--model_repository_path",
            "/models",
            "--model_name",
            WHISPER_MODEL_NAME,
            "--target_device",
            TARGET_DEVICE,
            "--task",
            "speech2text",
        ]
    )
    git_dir = MODEL_REPOSITORY / "OpenVINO" / WHISPER_MODEL_NAME / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)


def main() -> None:
    validate_target_device()
    ensure_tool(DOCKER_CMD[0])
    export_tinyllama()
    write_base_config()
    prepare_whisper()
    print(f"[done] Prepared OVMS repository in {MODEL_REPOSITORY}")
    print(f"[done] Target device: {TARGET_DEVICE}")
    print(f"[done] Docker command: {' '.join(DOCKER_CMD)}")
    print("Next: wsl docker compose up -d ovms")


if __name__ == "__main__":
    main()

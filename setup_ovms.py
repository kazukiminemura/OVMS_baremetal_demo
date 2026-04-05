#!/usr/bin/env python3
"""
Prepare a single OVMS model repository with TinyLlama and Whisper.

Requirements:
  pip install "optimum[openvino]>=1.18"
  pip install "huggingface_hub>=0.23"
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPOSITORY = Path("models")
TINYLLAMA_DIR = MODEL_REPOSITORY / "tinyllama"
WHISPER_DIR = MODEL_REPOSITORY / "OpenVINO" / "whisper-base-fp16-ov"
CONFIG_PATH = MODEL_REPOSITORY / "config.json"
WHISPER_SOURCE_MODEL = "OpenVINO/whisper-base-fp16-ov"
WHISPER_MODEL_NAME = "whisper-base-fp16-ov"
TARGET_DEVICE = os.environ.get("OVMS_TARGET_DEVICE", "GPU").upper()

TINYLLAMA_REQUIRED_FILES = [
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
]
WHISPER_REQUIRED_FILES = [
    "openvino_encoder_model.xml",
    "openvino_encoder_model.bin",
    "openvino_decoder_model.xml",
    "openvino_decoder_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "graph.pbtxt",
]

TINYLLAMA_GRAPH_TEMPLATE = """input_stream: "HTTP_REQUEST_PAYLOAD:input"
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
      models_path: "__MODEL_PATH__",
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


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PYTHONUTF8": "1"})
    if result.returncode != 0:
        sys.exit(result.returncode)


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


def validate_target_device() -> None:
    if TARGET_DEVICE not in {"CPU", "GPU"}:
        print("Error: OVMS_TARGET_DEVICE must be CPU or GPU.")
        sys.exit(1)


def normalize_path(path: Path) -> str:
    return path.resolve().as_posix()


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

    tinyllama_graph = (
        TINYLLAMA_GRAPH_TEMPLATE.replace("__MODEL_PATH__", normalize_path(TINYLLAMA_DIR))
        .replace("__TARGET_DEVICE__", TARGET_DEVICE)
    )
    (TINYLLAMA_DIR / "graph.pbtxt").write_text(tinyllama_graph, encoding="utf-8")


def write_base_config() -> None:
    MODEL_REPOSITORY.mkdir(parents=True, exist_ok=True)
    config = {
        "model_config_list": [
            {
                "config": {
                    "name": "tinyllama",
                    "base_path": normalize_path(TINYLLAMA_DIR),
                }
            },
            {
                "config": {
                    "name": WHISPER_MODEL_NAME,
                    "base_path": normalize_path(WHISPER_DIR),
                }
            }
        ]
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")


def prepare_whisper() -> None:
    if has_required_files(WHISPER_DIR, WHISPER_REQUIRED_FILES):
        print("[skip] Whisper is already downloaded")
        return

    WHISPER_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=WHISPER_SOURCE_MODEL,
        local_dir=str(WHISPER_DIR),
        local_dir_use_symlinks=False,
    )

    git_dir = WHISPER_DIR / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)


def main() -> None:
    validate_target_device()
    export_tinyllama()
    prepare_whisper()
    write_base_config()
    print(f"[done] Prepared OVMS repository in {MODEL_REPOSITORY}")
    print(f"[done] Target device: {TARGET_DEVICE}")
    print(f"[done] OVMS config: {CONFIG_PATH.resolve()}")
    print("Next: run `./start_ovms.ps1` after sourcing your OVMS setupvars script.")


if __name__ == "__main__":
    main()

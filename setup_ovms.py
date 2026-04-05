#!/usr/bin/env python3
"""
Prepare a single OVMS model repository with multiple LLMs and Whisper.

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

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    print("Installing missing dependency: huggingface_hub", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download


WHISPER_GRAPH_TEMPLATE = """input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node {
  name: "S2tExecutor"
  input_side_packet: "STT_NODE_RESOURCES:s2t_servable"
  calculator: "S2tCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
    [type.googleapis.com / mediapipe.S2tCalculatorOptions]: {
      models_path: "./",
      plugin_config: '{ "NUM_STREAMS": "1" }',
      target_device: "__TARGET_DEVICE__"
    }
  }
}
"""


def cleanup_hf_cache(model_dir: str | Path) -> None:
    model_path = Path(model_dir)

    cache_dir = model_path / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def ensure_whisper_speech2text_layout(model_dir: str | Path) -> None:
    """
    Arrange the Whisper snapshot for OVMS `speech2text` graph serving.

    Unlike classic OVMS model serving, the speech2text MediaPipe graph expects the
    Whisper OpenVINO files directly in the model directory next to `graph.pbtxt`.
    """
    model_path = Path(model_dir)
    cleanup_hf_cache(model_path)

    version_path = model_path / "1"
    if version_path.is_dir():
        for path in version_path.iterdir():
            target = model_path / path.name
            if not target.exists():
                shutil.move(str(path), str(target))
        shutil.rmtree(version_path, ignore_errors=True)

    graph_path = model_path / "graph.pbtxt"
    graph_path.write_text(
        WHISPER_GRAPH_TEMPLATE.replace("__TARGET_DEVICE__", TARGET_DEVICE),
        encoding="utf-8",
    )

MODEL_REPOSITORY = Path("models")
WHISPER_DIR = MODEL_REPOSITORY / "OpenVINO" / "whisper-base-fp16-ov"
CONFIG_PATH = MODEL_REPOSITORY / "config.json"
WHISPER_SOURCE_MODEL = "OpenVINO/whisper-base-fp16-ov"
WHISPER_MODEL_NAME = "whisper-base-fp16-ov"
TARGET_DEVICE = os.environ.get("OVMS_TARGET_DEVICE", "GPU").upper()
REQUESTED_LLM_MODELS = {
    name.strip().lower()
    for name in os.environ.get(
        "OVMS_LLM_MODELS",
        "tinyllama,qwen3-4b,llama-3.2-3b-instruct",
    ).split(",")
    if name.strip()
}

LLM_MODELS = [
    {
        "name": "tinyllama",
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "directory": MODEL_REPOSITORY / "tinyllama",
        "extra_export_args": [],
    },
    {
        "name": "qwen3-4b",
        "repo_id": "Qwen/Qwen3-4B",
        "directory": MODEL_REPOSITORY / "qwen3-4b",
        "extra_export_args": ["--trust-remote-code"],
    },
    {
        "name": "llama-3.2-3b-instruct",
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "directory": MODEL_REPOSITORY / "llama-3.2-3b-instruct",
        "extra_export_args": [],
    },
]

LLM_REQUIRED_FILES = [
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
        raise subprocess.CalledProcessError(result.returncode, cmd)


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


def get_enabled_llm_models() -> list[dict[str, object]]:
    available_names = {str(model["name"]) for model in LLM_MODELS}
    unknown_names = sorted(REQUESTED_LLM_MODELS - available_names)
    if unknown_names:
        print(f"Error: unknown OVMS_LLM_MODELS entries: {', '.join(unknown_names)}")
        sys.exit(1)
    return [model for model in LLM_MODELS if str(model["name"]) in REQUESTED_LLM_MODELS]


def normalize_path(path: Path) -> str:
    return path.resolve().as_posix()


def has_required_files(model_dir: Path, required_files: list[str]) -> bool:
    return model_dir.is_dir() and all((model_dir / required).exists() for required in required_files)


def export_llm_model(model: dict[str, object], cli: str) -> None:
    model_name = str(model["name"])
    repo_id = str(model["repo_id"])
    model_dir = Path(model["directory"])
    extra_export_args = [str(arg) for arg in model.get("extra_export_args", [])]

    if has_required_files(model_dir, LLM_REQUIRED_FILES):
        print(f"[skip] {model_name} is already exported")
    else:
        MODEL_REPOSITORY.mkdir(parents=True, exist_ok=True)
        try:
            run(
                [
                    cli,
                    "export",
                    "openvino",
                    "-m",
                    repo_id,
                    "--weight-format",
                    "int8",
                    "--task",
                    "text-generation-with-past",
                    *extra_export_args,
                    str(model_dir),
                ]
            )
        except subprocess.CalledProcessError as exc:
            print(f"Error: failed to export {model_name} from {repo_id}.")
            if repo_id.startswith("meta-llama/"):
                print("This is a gated Hugging Face repo.")
                print("Grant access on Hugging Face, then run `huggingface-cli login`.")
                print("If you want to skip Llama for now, set:")
                print('  $env:OVMS_LLM_MODELS="tinyllama,qwen3-4b"')
            raise SystemExit(exc.returncode) from exc

    llm_graph = (
        TINYLLAMA_GRAPH_TEMPLATE.replace("__MODEL_PATH__", normalize_path(model_dir))
        .replace("__TARGET_DEVICE__", TARGET_DEVICE)
    )
    (model_dir / "graph.pbtxt").write_text(llm_graph, encoding="utf-8")


def export_llm_models() -> None:
    cli = find_optimum_cli()
    for model in get_enabled_llm_models():
        export_llm_model(model, cli)


def write_base_config() -> None:
    MODEL_REPOSITORY.mkdir(parents=True, exist_ok=True)
    model_config_list = [
        {
            "config": {
                "name": str(model["name"]),
                "base_path": normalize_path(Path(model["directory"])),
            }
        }
        for model in get_enabled_llm_models()
    ]
    model_config_list.append(
        {
            "config": {
                "name": WHISPER_MODEL_NAME,
                "base_path": normalize_path(WHISPER_DIR),
                "graph_path": "graph.pbtxt",
            }
        }
    )
    config = {"model_config_list": model_config_list}
    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")


def prepare_whisper() -> None:
    WHISPER_DIR.parent.mkdir(parents=True, exist_ok=True)
    if has_required_files(WHISPER_DIR, WHISPER_REQUIRED_FILES):
        print("[skip] Whisper is already downloaded")
    else:
        downloaded_model_path = snapshot_download(
            repo_id=WHISPER_SOURCE_MODEL,
            local_dir=str(WHISPER_DIR),
            local_dir_use_symlinks=False,
        )
        cleanup_hf_cache(downloaded_model_path)

    git_dir = WHISPER_DIR / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)

    ensure_whisper_speech2text_layout(WHISPER_DIR)


def main() -> None:
    validate_target_device()
    export_llm_models()
    prepare_whisper()
    write_base_config()
    print(f"[done] Prepared OVMS repository in {MODEL_REPOSITORY}")
    print(f"[done] Target device: {TARGET_DEVICE}")
    print(f"[done] OVMS config: {CONFIG_PATH.resolve()}")
    print("Next: run `./start_ovms.ps1` after sourcing your OVMS setupvars script.")


if __name__ == "__main__":
    main()

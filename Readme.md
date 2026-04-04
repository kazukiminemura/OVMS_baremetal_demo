# OVMS Docker Demo - Whisper + TinyLlama

OpenVINO Model Server (OVMS) only demo with a single OVMS container.

```text
Audio (mic / WAV)
    |
    v
Whisper        -> OVMS (`/v3/audio/transcriptions`, port 8000)
    |
    v
TinyLlama      -> OVMS (`/v3/chat/completions`, port 8000)
    |
    v
Response
```

## Setup

### 1. Prepare OVMS model repository

```bash
python setup_ovms.py
```

This prepares a single `models/` repository for OVMS:

- exports `TinyLlama/TinyLlama-1.1B-Chat-v1.0` into `models/tinyllama/`
- pulls `OpenVINO/whisper-base-fp16-ov` into `models/whisper-base-fp16-ov/`
- creates `models/config.json` for both models

### 2. Start OVMS

```bash
docker compose up -d ovms
```

The single `ovms` service serves both TinyLlama and Whisper on port `8000`.
It is configured for GPU execution using the OVMS GPU image and `/dev/dri` passthrough.

### 3. Install Python packages

```bash
pip install -r requirements.txt
```

## Run

```bash
# microphone mode
python demo.py

# WAV file mode
python demo.py audio.wav

# text-only mode
python demo.py --text
```

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | Single OVMS service |
| `demo.py` | Client app using OVMS OpenAI-compatible APIs |
| `setup_ovms.py` | Prepares TinyLlama, Whisper, and `config.json` for OVMS |
| `requirements.txt` | Python runtime dependencies |

## Notes

- TinyLlama runs through OVMS chat completions.
- Whisper runs through OVMS audio transcriptions.
- Both models are mounted into one OVMS container via `./models:/models`.
- GPU use requires a host/container runtime where Intel GPU devices are exposed to the container.

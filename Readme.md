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

`setup_ovms.py` prepares OVMS models for `GPU` by default.
Docker commands are executed in WSL by default via `wsl docker`.

To force CPU-targeted models explicitly:

```bash
set OVMS_TARGET_DEVICE=CPU
python setup_ovms.py
```

This prepares a single `models/` repository for OVMS:

- exports `TinyLlama/TinyLlama-1.1B-Chat-v1.0` into `models/tinyllama/`
- pulls `OpenVINO/whisper-base-fp16-ov` into `models/whisper-base-fp16-ov/`
- creates `models/config.json` for both models

### 2. Start OVMS

```bash
wsl docker compose up -d ovms
```

The single `ovms` service serves both TinyLlama and Whisper on port `8000`.
The OVMS container uses the GPU-capable image. Actual model target device is chosen when running `setup_ovms.py`.
For WSL GPU execution, the compose file exposes `/dev/dxg` and mounts `/usr/lib/wsl` into the container.

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

If you change `OVMS_TARGET_DEVICE` or regenerate models, restart OVMS:

```bash
wsl docker compose restart ovms
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
- GPU mode assumes Docker is running inside WSL with GPU support available through `/dev/dxg`.

## Troubleshooting

- If `python demo.py` prints `OVMS connected ... []`, OVMS is running but failed to load the models. Re-run `python setup_ovms.py` and then `wsl docker compose restart ovms`.
- If OVMS logs show `Available devices for Open VINO: CPU`, GPU is not exposed to the container. Set `OVMS_TARGET_DEVICE=CPU`, rerun `python setup_ovms.py`, and restart OVMS.
- If you want to inspect OVMS errors directly, run:

```bash
wsl docker compose logs ovms --tail=200
```

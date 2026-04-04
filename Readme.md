# OVMS Docker Demo - Whisper + TinyLlama

OpenVINO Model Server (OVMS) only demo.

```text
Audio (mic / WAV)
    |
    v
Whisper        -> OVMS (`/v3/audio/transcriptions`, port 8001)
    |
    v
TinyLlama      -> OVMS (`/v3/chat/completions`, port 8000)
    |
    v
Response
```

## Setup

### 1. Prepare TinyLlama

```bash
python setup.py
```

`models/tinyllama/` is exported in OpenVINO format.

### 2. Start OVMS

```bash
docker compose --profile serve up -d ovms ovms-whisper
```

`ovms` serves TinyLlama on port `8000`.
`ovms-whisper` serves Whisper on port `8001`.

The Whisper container pulls `OpenVINO/whisper-base-fp16-ov` on first start, so the first boot can take a while.

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
| `docker-compose.yml` | OVMS services for TinyLlama and Whisper |
| `demo.py` | Client app using OVMS OpenAI-compatible APIs |
| `setup.py` | TinyLlama export script |
| `requirements.txt` | Python runtime dependencies |

## Notes

- TinyLlama runs through OVMS chat completions.
- Whisper runs through OVMS audio transcriptions.
- `models/whisper/` is no longer used by `demo.py`.

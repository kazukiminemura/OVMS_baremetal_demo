# OVMS Baremetal Demo - Whisper + OpenVINO LLMs

This repository runs OpenVINO Model Server (OVMS) on baremetal.
The setup follows the official OpenVINO 2026 baremetal deployment guide:
https://docs.openvino.ai/2026/model-server/ovms_docs_deploying_server_baremetal.html

```text
Audio (mic / WAV)
    |
    v
Whisper        -> OVMS (`/v3/audio/transcriptions`, port 8000)
    |
    v
LLM            -> OVMS (`/v3/chat/completions`, port 8000)
    |
    v
Response
```

## 1. Install OVMS on Windows 11

The official guide states that baremetal OVMS for Windows requires Microsoft Visual C++ Redistributable.

Download the package with Python support:

```powershell
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2026.0/ovms_windows_python_on.zip -o ovms.zip
tar -xf ovms.zip
```

This extracts an `ovms\` directory in the repository root.

Before starting OVMS in a new PowerShell session, load the environment variables:

```powershell
.\ovms\setupvars.ps1
```

## 2. Install Python packages

```powershell
pip install -r requirements.txt
```

`setup_ovms.py` uses:

- `optimum-cli` to export the LLMs to OpenVINO
- `huggingface_hub` to download the Whisper OpenVINO model repository

## 3. Prepare the OVMS model repository

```powershell
python setup_ovms.py
```

`setup_ovms.py` now:

- exports `TinyLlama/TinyLlama-1.1B-Chat-v1.0` into `models/tinyllama/`
- exports `Qwen/Qwen3-4B` into `models/qwen3-4b/`
- exports `meta-llama/Llama-3.2-3B-Instruct` into `models/llama-3.2-3b-instruct/`
- downloads `OpenVINO/whisper-base-fp16-ov` into `models/OpenVINO/whisper-base-fp16-ov/`
- creates `models/config.json` with host absolute paths for baremetal OVMS
- writes `graph.pbtxt` for each local LLM path with the selected target device

If you want to export `meta-llama/Llama-3.2-3B-Instruct`, make sure your Hugging Face account has access to the model and that you are authenticated locally, for example:

```powershell
huggingface-cli login
```

If you do not have Llama access yet, you can prepare only TinyLlama and Qwen:

```powershell
$env:OVMS_LLM_MODELS="tinyllama,qwen3-4b"
python setup_ovms.py
```

To force CPU-targeted LLM serving:

```powershell
$env:OVMS_TARGET_DEVICE="CPU"
python setup_ovms.py
```

Default is `GPU`.

## 4. Start OVMS on baremetal

After `.\ovms\setupvars.ps1` has been executed:

```powershell
.\start_ovms.ps1
```

This starts OVMS with:

```powershell
ovms --config_path .\models\config.json --rest_port 8000
```

## 5. Run the demo

```powershell
# microphone mode
python demo.py

# WAV file mode
python demo.py audio.wav

# text-only mode
python demo.py --text

# choose an exported LLM
python demo.py --model qwen3-4b
python demo.py --model llama-3.2-3b-instruct
```

## Files

| File | Purpose |
|---|---|
| `demo.py` | Client app using OVMS OpenAI-compatible APIs |
| `setup_ovms.py` | Prepares TinyLlama, Qwen3, Llama 3.2, Whisper, and `config.json` for baremetal OVMS |
| `start_ovms.ps1` | Starts local OVMS with the generated config |
| `requirements.txt` | Python runtime dependencies |

## Troubleshooting

- If `python demo.py` prints `OVMS connected ... []`, OVMS started but the models did not load. Run `python setup_ovms.py` again and restart `.\start_ovms.ps1`.
- If `.\start_ovms.ps1` cannot find `ovms`, run `.\ovms\setupvars.ps1` in the current PowerShell session first.
- If GPU execution is unavailable, set `$env:OVMS_TARGET_DEVICE="CPU"` and rerun `python setup_ovms.py`.

# OVMS Docker Demo — Whisper + TinyLlama

OpenVINO Model Server (OVMS) を使った音声対話デモ。

```
音声 (マイク/WAV)
    │
    ▼
Whisper base  ← OpenVINO GenAI (ローカル実行)
    │ テキスト
    ▼
TinyLlama     ← OVMS (Docker)  /v3/chat/completions
    │
    ▼
  応答出力
```

---

## セットアップ

### 1. モデル変換（初回のみ・数分かかります）

```bash
docker compose run --rm prep
```

`models/tinyllama/` と `models/whisper/` に OpenVINO 形式で保存されます。

### 2. OVMS を起動

```bash
docker compose --profile serve up -d ovms
```

### 3. Python 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

---

## デモ実行

```bash
# マイク入力モード
python demo.py

# WAVファイル指定
python demo.py audio.wav

# テキスト入力モード（マイクなし）
python demo.py --text
```

---

## 構成ファイル

| ファイル | 役割 |
|---|---|
| `docker-compose.yml` | モデル変換 + OVMS 起動 |
| `demo.py` | デモ本体（約60行） |
| `requirements.txt` | Python 依存パッケージ |

## ポイント

- **OVMS**: TinyLlama を OpenAI 互換 API (`/v3/chat/completions`) で提供
- **openvino_genai**: Whisper をローカルで実行（前処理を含め1行で動作）
- **int8 量子化**: TinyLlama を int8 に量子化してメモリ削減

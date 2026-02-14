# Qwen3-TTS 1.7B Chutes Deployment Reference

## Build & Deploy

```bash
# Build image (private)
chutes build chutes-tts-deployment/deploy_qwen_tts.py:chute --wait

# Verify image status
chutes images list --name qwen3-tts-1.7b

# Deploy chute
chutes deploy chutes-tts-deployment/deploy_qwen_tts.py:chute --accept-fee

# Warmup (after deploy)
chutes warmup qwen3-tts-1.7b
```

## API Reference

### POST /speak

Generate speech from text with voice cloning.

**Request body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | Text to synthesize |
| `language` | string | no | Language (default: `"English"`) |
| `ref_audio_b64` | string | no | Base64-encoded reference audio (WAV, 3+ seconds) |
| `ref_audio_url` | string | no | URL to reference audio |
| `ref_text` | string | no | Transcript of the reference audio |

Either `ref_audio_b64` or `ref_audio_url` is required.

**Response:** `audio/wav` binary

**Example:**
```bash
REF_B64=$(base64 -w0 reference_audio.wav)

curl -X POST https://api.chutes.ai/v1/chutes/<chute-id>/speak \
  -H "Authorization: Bearer $CHUTES_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Hello, this is a test of voice cloning.\",
    \"language\": \"English\",
    \"ref_audio_b64\": \"$REF_B64\",
    \"ref_text\": \"Transcript of the reference audio\"
  }" --output test_output.wav
```

## Key Dependencies

Installed via `qwen-tts`:
- `transformers==4.57.3`
- `accelerate==1.12.0`
- `soundfile`

Installed separately:
- `torch` (latest)
- `torchaudio` (latest)

System packages: `libsndfile1`, `sox`, `ffmpeg`, `git`, `git-lfs`, `curl`

## Supported Languages

English, Chinese, Japanese, Korean, French, German, Spanish, and other languages supported by the Qwen3-TTS model.

## Troubleshooting

**Build fails at pip install:**
Check that the base image `parachutes/base-python:3.12.9` is available and has CUDA support.

**Model download fails during build:**
Ensure the build environment has internet access. The model (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`) and tokenizer (`Qwen/Qwen3-TTS-Tokenizer-12Hz`) are downloaded from HuggingFace Hub.

**Runtime OOM:**
The model requires ~16GB VRAM. Ensure `min_vram_gb_per_gpu=16` in the NodeSelector.

**sox/audio processing errors:**
Verify `sox` and `libsndfile1` are installed in the image (included in the apt-get step).

**Slow first request:**
Expected behavior — the warmup step was removed to avoid invalid zero-audio errors. First request takes ~1-2s longer.

**Checking logs:**
```bash
chutes logs <chute-id>
```

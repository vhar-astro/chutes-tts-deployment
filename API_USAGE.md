# Qwen3-TTS 1.7B Voice Cloning API

## Endpoint

```
POST https://letternumber123-qwen3-tts-1-7b.chutes.ai/speak
```

## Authentication

```
Authorization: Bearer $CHUTES_API_TOKEN
```

## Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `language` | string | No | Language (default: `"English"`) |
| `ref_audio_b64` | string | One of b64/url required | Base64-encoded reference audio (WAV) |
| `ref_audio_url` | string | One of b64/url required | Public URL to reference audio |
| `ref_text` | string | No | Transcript of reference audio (improves quality) |

You must provide either `ref_audio_b64` or `ref_audio_url`. If `ref_text` is provided, the model uses higher-quality ICL mode. Without it, it uses x-vector speaker embedding mode.

## Response

- **Content-Type**: `audio/wav`
- **Body**: Raw WAV audio bytes (24kHz, mono, 16-bit)

## Examples

### Minimal (base64 audio, no transcript)

```bash
curl -X POST \
  https://letternumber123-qwen3-tts-1-7b.chutes.ai/speak \
  -H "Authorization: Bearer $CHUTES_API_TOKEN" \
  -H "Content-Type: application/json" \
  -o output.wav \
  -d '{
    "text": "Hello, this is a voice cloning test.",
    "ref_audio_b64": "<base64-encoded-wav-data>"
  }'
```

### With URL + transcript (best quality)

```bash
curl -X POST \
  https://letternumber123-qwen3-tts-1-7b.chutes.ai/speak \
  -H "Authorization: Bearer $CHUTES_API_TOKEN" \
  -H "Content-Type: application/json" \
  -o output.wav \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "language": "English",
    "ref_audio_url": "https://example.com/reference-voice.wav",
    "ref_text": "This is the transcript of what is said in the reference audio."
  }'

```

### Python example

```python
import requests
import base64

API_URL = "https://letternumber123-qwen3-tts-1-7b.chutes.ai/speak"
API_TOKEN = "your-chutes-api-token"

# Load and encode reference audio
with open("reference_voice.wav", "rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    },
    json={
        "text": "Hello, this is a voice cloning test.",
        "language": "English",
        "ref_audio_b64": ref_audio_b64,
        "ref_text": "Optional transcript of what is said in the reference audio.",
    },
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print(f"Saved {len(response.content)} bytes to output.wav")
else:
    print(f"Error {response.status_code}: {response.text}")
```

### Encode a local WAV to base64 (helper)

```bash
base64 -w 0 reference_voice.wav > ref_audio_b64.txt
```

## Notes

- Reference audio should be at least 3 seconds of clear speech
- Providing `ref_text` (transcript of the reference audio) significantly improves cloning quality
- Supported languages: English, Chinese, and others
- The chute must be "hot" (warmed up) to respond; cold starts may timeout

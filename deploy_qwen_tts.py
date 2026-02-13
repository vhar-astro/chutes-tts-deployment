import os
import uuid
from io import BytesIO
from typing import Optional
from loguru import logger
from fastapi import Response
from pydantic import BaseModel, Field
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

# Define the custom Docker image
image = (
    Image(
        username="letternumber123",
        name="qwen3-tts-1.7b",
        tag="0.0.12",
        readme="## Qwen3-TTS 1.7B Base\n\nText-to-speech with voice cloning capabilities using Qwen/Qwen3-TTS-12Hz-1.7B-Base.",
    )
    .from_base("parachutes/base-python:3.12.9")
    # 1. Install system dependencies as root
    .set_user("root")
    .run_command("apt-get update && apt-get install -y libsndfile1 git git-lfs curl")
    
    # 2. Install Python dependencies as chutes user
    .set_user("chutes")
    # First batch: ensures psutil, torch, and packaging are present for flash-attn build
    .run_command("pip install --no-cache-dir psutil torch torchaudio packaging")
    # Second: build flash-attn using the environment we just prepared
    .run_command("pip install flash-attn --no-build-isolation")
    # Third: the application itself
    .run_command("pip install qwen-tts soundfile transformers accelerate")
)

# Chute definition
chute = Chute(
    username="chutes",
    name="qwen3-tts-1.7b",
    tagline="Qwen3-TTS 1.7B Base - Voice Cloning TTS",
    readme="## Qwen3-TTS 1.7B Base\n\nText-to-speech with voice cloning capabilities using Qwen/Qwen3-TTS-12Hz-1.7B-Base.",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16), 
    concurrency=1,
    allow_external_egress=True,
)

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: str = Field("English", description="Language of the text")
    ref_audio_b64: Optional[str] = Field(None, description="Base64 encoded reference audio")
    ref_audio_url: Optional[str] = Field(None, description="URL to reference audio")
    ref_text: Optional[str] = Field(None, description="Transcript of the reference audio")

@chute.on_startup()
async def initialize(self):
    """
    Initialize model and dependencies.
    """
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    import tempfile

    logger.info("Loading Qwen3-TTS model...")
    self.model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    self.sf = sf
    self.torch = torch
    
    # Warmup pass
    logger.info("Running warmup generation...")
    try:
        dummy_wav = BytesIO()
        zero_data = torch.zeros(16000).numpy() 
        sf.write(dummy_wav, zero_data, 16000, format='WAV')
        dummy_wav.seek(0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
             f.write(dummy_wav.read())
             dummy_path = f.name
        
        try:
            self.model.generate_voice_clone(
                text="Warmup complete.",
                language="English",
                ref_audio=dummy_path,
            )
            logger.success("Warmup successful!")
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
    except Exception as e:
        logger.warning(f"Warmup generation failed (non-critical): {e}")

@chute.cord(
    public_api_path="/speak",
    public_api_method="POST",
    stream=False,
    output_content_type="audio/wav",
)
async def speak(self, args: TTSRequest) -> Response:
    """
    Generate speech from text.
    """
    import base64
    import tempfile
    
    temp_ref_file = None
    final_ref_audio = args.ref_audio_url
    
    if args.ref_audio_b64:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(base64.b64decode(args.ref_audio_b64))
            temp_ref_file = f.name
        final_ref_audio = temp_ref_file
    
    if not final_ref_audio:
         return Response(content="Reference audio (b64 or url) is required for voice cloning.", status_code=400)

    try:
        # Generate voice
        wavs, sr = self.model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=final_ref_audio,
            ref_text=args.ref_text,
        )
        
        # Save output to buffer
        buffer = BytesIO()
        self.sf.write(buffer, wavs[0], sr, format='WAV')
        buffer.seek(0)
        
        logger.info(f"Generated {buffer.getbuffer().nbytes} bytes of audio")
        
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={uuid.uuid4()}.wav",
            },
        )
    finally:
        if temp_ref_file and os.path.exists(temp_ref_file):
            os.remove(temp_ref_file)

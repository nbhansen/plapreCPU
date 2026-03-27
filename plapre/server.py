"""
FastAPI server for PlapreCPU Danish TTS with chunked PCM streaming (CPU-only).

Start with:
    plapre-cpu-serve --port 8000

Or:
    uvicorn plapre.server:app
"""

import asyncio
import logging
import os
import struct
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from plapre.inference import SAMPLE_RATE, Plapre

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_tts: Plapre | None = None
_vocoder_sem: asyncio.Semaphore | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts, _vocoder_sem

    checkpoint = os.environ.get("PLAPRE_CHECKPOINT", "syvai/plapre-nano")
    quant = os.environ.get("PLAPRE_QUANT", "q8_0")
    n_threads = int(os.environ.get("PLAPRE_THREADS", "4"))
    n_ctx = int(os.environ.get("PLAPRE_CTX", "2048"))
    log.info("Loading model %s (quant=%s, threads=%d, ctx=%d) …",
             checkpoint, quant, n_threads, n_ctx)
    _tts = Plapre(
        checkpoint=checkpoint,
        quant=quant,
        n_threads=n_threads,
        n_ctx=n_ctx,
    )
    # Serialize vocoder calls to avoid concurrent heavy CPU work
    _vocoder_sem = asyncio.Semaphore(1)
    log.info("Model ready.")
    yield
    _tts = None


app = FastAPI(title="PlapreCPU TTS", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    text: str
    speaker: str | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to 16-bit signed LE PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        spk = _tts._resolve_speaker(req.speaker, None, None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    gen_kwargs = dict(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )

    sentences = _tts._split_sentences(req.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No text provided")

    silence_samples = int(0.1 * SAMPLE_RATE)
    silence_bytes = struct.pack(f"<{silence_samples}h", *([0] * silence_samples))

    async def generate():
        for i, sent in enumerate(sentences):
            log.info("Generating sentence %d/%d: %s", i + 1, len(sentences), sent)
            async with _vocoder_sem:
                audio = await asyncio.to_thread(
                    _tts._generate_audio, sent, spk, **gen_kwargs
                )
            if audio is not None:
                yield _float32_to_pcm16(audio)
                if i < len(sentences) - 1:
                    yield silence_bytes

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


@app.get("/v1/speakers")
async def speakers():
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"speakers": list(_tts.speakers.keys())}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="PlapreCPU TTS server (no CUDA required)")
    parser.add_argument(
        "--checkpoint", default="syvai/plapre-nano",
        help="HuggingFace checkpoint (default: syvai/plapre-nano)",
    )
    parser.add_argument("--quant", default="q8_0", help="GGUF quantization (default: q8_0)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length (default: 2048)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    os.environ["PLAPRE_CHECKPOINT"] = args.checkpoint
    os.environ["PLAPRE_QUANT"] = args.quant
    os.environ["PLAPRE_THREADS"] = str(args.threads)
    os.environ["PLAPRE_CTX"] = str(args.ctx)
    uvicorn.run(
        "plapre.server:app",
        host=args.host,
        port=args.port,
        http="httptools",
    )


if __name__ == "__main__":
    main()

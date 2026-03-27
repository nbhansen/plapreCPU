"""
PlapreCPU – Danish TTS inference using llama.cpp for CPU-only generation.

Usage:
    from plapre import Plapre

    tts = Plapre("syvai/plapre-nano")
    tts.speak("Hej, hvordan har du det?", output="output.wav")

    # Voice cloning
    tts.speak("Hej", output="cloned.wav", speaker_wav="reference.wav")

    # Long text with sentence splitting
    tts.speak("Sætning et. Sætning to.", output="long.wav", split_sentences=True)
"""

import ctypes
import json
import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import llama_cpp
from llama_cpp import (
    llama_batch_free,
    llama_batch_init,
    llama_decode,
    llama_get_memory,
    llama_memory_clear,
    llama_sampler_chain_add,
    llama_sampler_chain_init,
    llama_sampler_chain_default_params,
    llama_sampler_free,
    llama_sampler_init_dist,
    llama_sampler_init_temp,
    llama_sampler_init_top_k,
    llama_sampler_init_top_p,
    llama_sampler_sample,
)

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SPEAKER_DIM = 128
HIDDEN_SIZE = 960
KANADE_MODEL = "frothywater/kanade-25hz-clean"

GGUF_QUANTS = ["f16", "q8_0", "q6_k", "q4_k_m", "q4_0"]
DEFAULT_QUANT = "q8_0"


class Plapre:
    """Danish text-to-speech synthesis – CPU-only, no CUDA required."""

    def __init__(
        self,
        checkpoint: str = "syvai/plapre-nano",
        quant: str = DEFAULT_QUANT,
        n_ctx: int = 2048,
        n_threads: int = 4,
        device: str | None = None,
    ):
        self.device = torch.device("cpu")
        self._checkpoint = checkpoint

        # --- Tokenizer (CPU) ---
        log.info("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.audio_token_start = self.tokenizer.convert_tokens_to_ids("<audio_0>")
        self.audio_token_end = self.tokenizer.convert_tokens_to_ids("<audio_12799>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("</audio>")
        self.text_tag = self.tokenizer.convert_tokens_to_ids("<text>")
        self.audio_tag = self.tokenizer.convert_tokens_to_ids("<audio>")
        self.eos_id = self.tokenizer.eos_token_id

        # --- Speaker projection (CPU, float32) ---
        self.speaker_proj = self._load_speaker_proj(checkpoint)

        # --- Resolve GGUF model ---
        gguf_path = self._resolve_gguf(checkpoint, quant)
        log.info("Using GGUF model: %s", gguf_path)

        # --- llama.cpp model (CPU-only) ---
        log.info("Loading llama.cpp model …")
        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = 0
        mparams.use_mmap = True

        self._model = llama_cpp.llama_model_load_from_file(
            gguf_path.encode("utf-8"), mparams,
        )
        if not self._model:
            raise RuntimeError(f"Failed to load GGUF model: {gguf_path}")

        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = 512
        cparams.n_ubatch = 512
        cparams.n_threads = n_threads
        cparams.n_threads_batch = n_threads
        cparams.flash_attn = False

        self._ctx = llama_cpp.llama_init_from_model(self._model, cparams)
        if not self._ctx:
            raise RuntimeError("Failed to create llama context")

        # --- Speakers (CPU) ---
        self.speakers = self._load_speakers()
        self.default_speaker = next(iter(self.speakers))
        log.info(
            "Loaded %d speaker(s): %s (default: %s)",
            len(self.speakers),
            list(self.speakers.keys()),
            self.default_speaker,
        )

        # --- Kanade vocoder (CPU) ---
        log.info("Loading Kanade vocoder …")
        from kanade_tokenizer import KanadeModel, load_vocoder

        self.kanade = KanadeModel.from_pretrained(KANADE_MODEL).eval().to(self.device)
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)

        # Cache for projected speaker embeddings
        self._proj_cache: dict[bytes, np.ndarray] = {}

        log.info("Ready – device=cpu, n_threads=%d", n_threads)

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            llama_cpp.llama_free(self._ctx)
        if hasattr(self, "_model") and self._model:
            llama_cpp.llama_model_free(self._model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(
        self,
        text: str,
        output: str = "output.wav",
        speaker: str | None = None,
        speaker_wav: str | None = None,
        speaker_emb: torch.Tensor | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 500,
        split_sentences: bool = False,
        silence_duration: float = 0.1,
    ) -> np.ndarray:
        """Synthesize speech and save to *output*. Returns the audio as a numpy array."""
        spk = self._resolve_speaker(speaker, speaker_wav, speaker_emb)

        gen_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        if split_sentences:
            sentences = self._split_sentences(text)  # normalize included
            log.info("Generating %d sentences", len(sentences))
            silence = np.zeros(
                int(silence_duration * SAMPLE_RATE), dtype=np.float32
            )
            chunks = []
            for i, sent in enumerate(sentences):
                log.info("Sentence %d/%d: %s", i + 1, len(sentences), sent)
                audio_chunk = self._generate_audio(sent, spk, **gen_kwargs)
                if audio_chunk is not None:
                    chunks.append(audio_chunk)
                    if i < len(sentences) - 1:
                        chunks.append(silence)
            if not chunks:
                log.error("No audio generated for any sentence.")
                return np.array([], dtype=np.float32)
            audio = np.concatenate(chunks)
        else:
            audio = self._generate_audio(
                self._normalize_text(text), spk, **gen_kwargs
            )
            if audio is None:
                log.error(
                    "No audio tokens generated. Try different temperature/top_p."
                )
                return np.array([], dtype=np.float32)

        sf.write(output, audio, SAMPLE_RATE)
        log.info("Saved %.2fs audio to %s", len(audio) / SAMPLE_RATE, output)
        return audio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gguf(checkpoint: str, quant: str) -> str:
        if quant not in GGUF_QUANTS:
            quant = DEFAULT_QUANT
        model_name = checkpoint.rstrip("/").split("/")[-1]
        repo_path = f"gguf/{model_name}.{quant}.gguf"
        local = Path(checkpoint) / repo_path
        if local.exists():
            return str(local)
        return hf_hub_download(checkpoint, repo_path)

    def _load_speakers(self) -> dict[str, torch.Tensor]:
        path = Path(__file__).parent / "speakers.json"
        with open(path) as f:
            raw = json.load(f)
        return {
            name: torch.tensor(emb, dtype=torch.float32)
            for name, emb in raw.items()
        }

    def _load_speaker_proj(self, checkpoint: str) -> nn.Linear:
        proj = nn.Linear(SPEAKER_DIM, HIDDEN_SIZE)
        local = Path(checkpoint) / "speaker_proj.pt"
        if local.exists():
            proj.load_state_dict(torch.load(local, map_location="cpu"))
        else:
            path = hf_hub_download(checkpoint, "speaker_proj.pt")
            proj.load_state_dict(torch.load(path, map_location="cpu"))
        return proj.float().eval()

    @torch.no_grad()
    def _project_speaker(self, speaker_emb: torch.Tensor) -> np.ndarray:
        """Project 128-dim speaker embedding to 960-dim hidden, returns float32 ndarray."""
        key = speaker_emb.cpu().float().numpy().tobytes()
        if key in self._proj_cache:
            return self._proj_cache[key]
        hidden = self.speaker_proj(speaker_emb.cpu().float())
        result = hidden.numpy().copy()
        self._proj_cache[key] = result
        return result

    def _build_prompt(self, text: str) -> list[int]:
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.text_tag] + text_ids + [self.audio_tag]

    def _generate_tokens(
        self,
        prompt_tokens: list[int],
        speaker_hidden: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> list[int]:
        """Generate audio token IDs using llama.cpp."""
        llama_memory_clear(llama_get_memory(self._ctx), True)

        n_prompt = len(prompt_tokens)

        # Decode speaker embedding at position 0
        embd_batch = llama_batch_init(1, HIDDEN_SIZE, 1)
        embd_batch.n_tokens = 1
        ctypes.memmove(embd_batch.embd, speaker_hidden.ctypes.data, HIDDEN_SIZE * 4)
        embd_batch.pos[0] = 0
        embd_batch.n_seq_id[0] = 1
        embd_batch.seq_id[0][0] = 0
        embd_batch.logits[0] = 0
        rc = llama_decode(self._ctx, embd_batch)
        llama_batch_free(embd_batch)
        if rc != 0:
            raise RuntimeError(f"Speaker embedding decode failed: {rc}")

        # Decode prompt tokens at positions 1..N
        batch = llama_batch_init(max(n_prompt, 512), 0, 1)
        batch.n_tokens = n_prompt
        for i, tid in enumerate(prompt_tokens):
            batch.token[i] = tid
            batch.pos[i] = i + 1
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1 if i == n_prompt - 1 else 0

        rc = llama_decode(self._ctx, batch)
        if rc != 0:
            llama_batch_free(batch)
            raise RuntimeError(f"Prompt decode failed: {rc}")

        # Sampler
        sparams = llama_sampler_chain_default_params()
        smpl = llama_sampler_chain_init(sparams)
        if top_k > 0:
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k))
        if top_p < 1.0:
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1))
        if temperature > 0:
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature))
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42))

        generated = []
        pos = 1 + n_prompt
        for _ in range(max_tokens):
            new_token = llama_sampler_sample(smpl, self._ctx, -1)
            if new_token == self.audio_end_id or new_token == self.eos_id:
                break
            generated.append(new_token)

            batch.n_tokens = 1
            batch.token[0] = new_token
            batch.pos[0] = pos
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1

            rc = llama_decode(self._ctx, batch)
            if rc != 0:
                break
            pos += 1

        llama_sampler_free(smpl)
        llama_batch_free(batch)
        return generated

    def _tokens_to_audio(
        self, tokens: list[int], speaker_emb: torch.Tensor
    ) -> np.ndarray | None:
        """Convert generated token IDs to audio waveform via Kanade + Vocos."""
        kanade_indices = [
            tid - self.audio_token_start
            for tid in tokens
            if self.audio_token_start <= tid <= self.audio_token_end
        ]
        if not kanade_indices:
            return None

        tokens_tensor = torch.tensor(
            kanade_indices, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            mel = self.kanade.decode(
                content_token_indices=tokens_tensor,
                global_embedding=speaker_emb.float().to(self.device),
            )
            from kanade_tokenizer import vocode

            waveform = vocode(self.vocoder, mel.unsqueeze(0))
        return waveform.squeeze().cpu().numpy()

    def _generate_audio(
        self,
        text: str,
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> np.ndarray | None:
        prompt_ids = self._build_prompt(text)
        speaker_hidden = self._project_speaker(speaker_emb)
        generated = self._generate_tokens(
            prompt_ids, speaker_hidden, temperature, top_p, top_k, max_tokens,
        )
        return self._tokens_to_audio(generated, speaker_emb)

    def _generate_audio_batch(
        self,
        texts: list[str],
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> list[np.ndarray | None]:
        """Generate audio for multiple texts sequentially."""
        return [
            self._generate_audio(t, speaker_emb, temperature, top_p, top_k, max_tokens)
            for t in texts
        ]

    def _extract_speaker_emb(self, wav_path: str) -> torch.Tensor:
        import torchaudio

        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T  # (channels, samples)
        wav = torch.from_numpy(data)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        with torch.no_grad():
            features = self.kanade.encode(wav.to(self.device))
        return features.global_embedding

    def _resolve_speaker(
        self,
        speaker: str | None,
        speaker_wav: str | None,
        speaker_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        if speaker_emb is not None:
            return speaker_emb.to(self.device)
        if speaker_wav is not None:
            emb = self._extract_speaker_emb(speaker_wav)
            log.info(
                "Speaker embedding from %s, norm=%.3f", speaker_wav, emb.norm()
            )
            return emb
        name = speaker or self.default_speaker
        if name not in self.speakers:
            raise ValueError(
                f"Unknown speaker '{name}'. Available: {list(self.speakers.keys())}"
            )
        return self.speakers[name]

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        """Replace numbers with Danish words (e.g. '2,1' → 'to komma et')."""
        from num2words import num2words

        def _replace(m):
            raw = m.group()
            try:
                return num2words(float(raw.replace(",", ".")), lang="da")
            except (ValueError, OverflowError):
                return raw

        return re.sub(r"\d+(?:[,\.]\d+)?", _replace, text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize raw article/document text for TTS."""
        # Remove trailing separators and image captions (e.g. "--- caption text")
        text = re.sub(r"\s*-{2,}.*$", "", text.strip(), flags=re.DOTALL)
        # Collapse whitespace (newlines, tabs, multiple spaces → single space)
        text = re.sub(r"\s+", " ", text)
        # Numbers → Danish words
        text = Plapre._normalize_numbers(text)
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Normalize first
        text = Plapre._normalize_text(text)
        # Split on sentence-ending punctuation followed by space,
        # but require 2+ word chars before the punctuation to avoid
        # splitting on abbreviations like "H.C." or "f."
        parts = re.split(r"(?<=\w{2}[.!?])\s+", text)
        result = []
        for p in parts:
            p = p.strip()
            # Strip leading dialogue dashes (Danish convention: "- quote")
            p = re.sub(r"^[-–—]\s+", "", p)
            if p:
                result.append(p)
        return result

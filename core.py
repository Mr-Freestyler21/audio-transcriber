"""
Core transcription logic shared by the CLI (transcribe.py) and the web UI (app.py).

Uses faster-whisper (CTranslate2 build of OpenAI Whisper). No PyTorch and no
external ffmpeg install are required: the `av` dependency decodes the audio.
"""
from __future__ import annotations

import os

# Silence a harmless Windows-only warning about symlink caching (set before the
# huggingface_hub import that faster_whisper triggers).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import ctranslate2
from faster_whisper import WhisperModel

# Model used when the caller doesn't specify one. Override with the WHISPER_MODEL
# environment variable, e.g.  set WHISPER_MODEL=medium
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")

# Offered in the CLI (--model) and the web UI dropdown, fastest -> most accurate.
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]


def pick_device() -> tuple[str, str]:
    """Return (device, compute_type). Uses the GPU if one is available, else CPU."""
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@lru_cache(maxsize=4)
def load_model(model_size: str) -> WhisperModel:
    """Load (and cache) a Whisper model.

    The first time a given model is used it is downloaded from Hugging Face and
    cached under ~/.cache/huggingface, so the first run of each size is slower.
    """
    device, compute_type = pick_device()
    return WhisperModel(model_size, device=device, compute_type=compute_type)


@dataclass
class Segment:
    start: float          # seconds
    end: float            # seconds
    text: str


@dataclass
class TranscriptionResult:
    text: str                       # full transcript as one block
    segments: list[Segment]         # time-stamped chunks
    language: str                   # detected (or forced) language code
    language_probability: float     # confidence of language detection (0-1)
    duration: float                 # audio length in seconds
    model: str


def transcribe(
    audio_path: str,
    model_size: str = DEFAULT_MODEL,
    language: Optional[str] = None,      # None => auto-detect
    *,
    beam_size: int = 5,                  # >1 = careful beam search (more accurate)
    vad_filter: bool = True,             # skip silence -> fewer hallucinations
    word_timestamps: bool = False,
    initial_prompt: Optional[str] = None,  # names/jargon hint to improve accuracy
    progress: Optional[Callable[[float, float], None]] = None,
) -> TranscriptionResult:
    """Transcribe one audio file and return the full text plus timed segments.

    `progress`, if given, is called as progress(seconds_done, seconds_total).
    """
    model = load_model(model_size)

    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
    )

    collected: list[Segment] = []
    raw_parts: list[str] = []
    # faster-whisper is lazy: the work happens as we iterate this generator.
    for seg in segments_gen:
        raw_parts.append(seg.text)
        collected.append(Segment(seg.start, seg.end, seg.text.strip()))
        if progress and info.duration:
            progress(min(seg.end, info.duration), info.duration)

    if progress and info.duration:
        progress(info.duration, info.duration)

    # Concatenating raw text preserves correct spacing for both spaced languages
    # (segments carry a leading space) and non-spaced ones (Chinese, Japanese...).
    full_text = "".join(raw_parts).strip()

    return TranscriptionResult(
        text=full_text,
        segments=collected,
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        model=model_size,
    )


# --------------------------------------------------------------------------- #
# Output formatters
# --------------------------------------------------------------------------- #
def _fmt_ts(seconds: float, *, srt: bool = False) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_timestamped_text(segments: list[Segment]) -> str:
    """Plain text with a [HH:MM:SS] marker before each segment."""
    lines = []
    for seg in segments:
        stamp = _fmt_ts(seg.start).split(".")[0]  # drop milliseconds
        lines.append(f"[{stamp}] {seg.text}")
    return "\n".join(lines)


def to_srt(segments: list[Segment]) -> str:
    """SubRip subtitle format."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n"
            f"{_fmt_ts(seg.start, srt=True)} --> {_fmt_ts(seg.end, srt=True)}\n"
            f"{seg.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    """WebVTT subtitle format."""
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n"
            f"{seg.text}\n"
        )
    return "\n".join(blocks)

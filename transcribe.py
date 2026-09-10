"""
Command-line transcription tool.

Examples
--------
  # Transcribe one file -> writes interview.txt next to it
  python transcribe.py interview.mp3

  # Higher accuracy model, force English, also write subtitles
  python transcribe.py interview.mp3 --model medium --language en --format all

  # Transcribe every audio file in a folder into ./out
  python transcribe.py ./recordings --output ./out

  # Add a hint of names/jargon to improve accuracy
  python transcribe.py lecture.m4a --prompt "Dr. Nwosu, mitochondria, ATP synthase"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import core

AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".m4b", ".flac", ".ogg", ".oga", ".opus",
    ".aac", ".wma", ".aiff", ".aif",
    # common video containers (audio track is extracted automatically)
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v",
}


def gather_inputs(paths: list[str]) -> list[Path]:
    """Expand any folders in `paths` into the audio/video files inside them."""
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(
                sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"  ! skipping (not found): {raw}", file=sys.stderr)
    return files


def _progress_printer(name: str):
    last = [0.0]

    def cb(done: float, total: float):
        now = time.time()
        if now - last[0] < 0.25 and done < total:
            return  # throttle redraws
        last[0] = now
        pct = (done / total * 100) if total else 0
        bar = "#" * int(pct // 4)
        print(f"\r  {name}: [{bar:<25}] {pct:5.1f}%", end="", flush=True)

    return cb


def write_outputs(result: core.TranscriptionResult, source: Path,
                  out_dir: Path, fmt: str, timestamps: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    written: list[Path] = []

    def _write(suffix: str, content: str):
        target = out_dir / f"{stem}{suffix}"
        target.write_text(content, encoding="utf-8")
        written.append(target)

    want = {"txt", "srt", "vtt"} if fmt == "all" else {fmt}
    if "txt" in want:
        body = core.to_timestamped_text(result.segments) if timestamps else result.text
        _write(".txt", body + "\n")
    if "srt" in want:
        _write(".srt", core.to_srt(result.segments))
    if "vtt" in want:
        _write(".vtt", core.to_vtt(result.segments))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Accurate offline audio/video -> text transcription (Whisper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+",
                        help="Audio/video file(s) or folder(s) to transcribe.")
    parser.add_argument("-m", "--model", default=core.DEFAULT_MODEL,
                        choices=core.MODEL_CHOICES,
                        help="Model size: bigger = more accurate but slower.")
    parser.add_argument("-l", "--language", default=None,
                        help="Language code (e.g. en, es, fr). Omit to auto-detect.")
    parser.add_argument("-o", "--output", default=None,
                        help="Folder to write transcripts into. "
                             "Default: next to each source file.")
    parser.add_argument("-f", "--format", default="txt",
                        choices=["txt", "srt", "vtt", "all"],
                        help="Output format(s).")
    parser.add_argument("--timestamps", action="store_true",
                        help="For txt output, prefix each line with a [HH:MM:SS] time.")
    parser.add_argument("--prompt", default=None,
                        help="Optional hint of names/jargon to bias transcription.")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam search width. Higher can be more accurate/slower.")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable voice-activity filtering (transcribe silence too).")
    args = parser.parse_args()

    files = gather_inputs(args.inputs)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 1

    device, compute = core.pick_device()
    print(f"Model: {args.model}  |  Device: {device} ({compute})  |  Files: {len(files)}")
    print("(The first run of a model downloads it once, then it's cached.)\n")

    failures = 0
    for source in files:
        print(f"- {source.name}")
        try:
            result = core.transcribe(
                str(source),
                model_size=args.model,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=not args.no_vad,
                initial_prompt=args.prompt,
                progress=_progress_printer(source.stem),
            )
        except Exception as exc:  # keep going with the remaining files
            print(f"\n  ! failed: {exc}", file=sys.stderr)
            failures += 1
            continue

        out_dir = Path(args.output) if args.output else source.parent
        written = write_outputs(result, source, out_dir, args.format, args.timestamps)
        conf = f"{result.language_probability:.0%}"
        print(f"\n  detected {result.language} ({conf}), {result.duration:.0f}s audio")
        for w in written:
            print(f"  -> {w}")
        print()

    if failures:
        print(f"Done with {failures} failure(s).", file=sys.stderr)
        return 2
    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

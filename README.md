# 🎧 Audio → Text Transcriber

Accurate, **fully offline** audio & video transcription powered by OpenAI's Whisper
model (via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)). No API keys,
no accounts, no per-minute cost — your audio never leaves your machine. Ships with
**both** a browser UI and a command-line tool.

## Features

- 🎯 **Accurate** — beam-search decoding + voice-activity filtering (skips silence,
  reduces made-up words).
- 🔒 **Private & offline** — runs locally; nothing is uploaded.
- 🌍 **Auto language detection** (or force a specific language).
- 🖥️ **Two interfaces** — a drag-and-drop browser app *and* a scriptable CLI.
- 📄 **Multiple outputs** — plain text, timestamped text, and subtitles (`.srt`, `.vtt`).
- 🎬 Works on **audio and video** files (the audio track is read automatically).
- ⚡ Uses your **GPU automatically** if one is available, otherwise runs on CPU.

## Requirements

- Python **3.11+** (developed and tested on 3.14)
- No separate ffmpeg install needed — audio decoding is bundled.
- No GPU required (a CUDA GPU is used automatically if present).

## Installation

```bash
git clone https://github.com/Mr-Freestyler21/audio-transcriber.git
cd audio-transcriber

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The first time you use a given model it downloads once (see the table below) and is
cached under `~/.cache/huggingface` for offline use afterwards.

## Usage

### Browser UI

```bash
python app.py
```

Opens `http://127.0.0.1:7860`. Upload an audio file (or record from your mic), pick a
model and language, click **Transcribe**, then copy or download `.txt` / `.srt` / `.vtt`.

> On Windows you can also just double-click **`run-web.bat`**.

### Command line

```bash
# Transcribe one file -> writes interview.txt next to it
python transcribe.py interview.mp3

# More accurate model, force English, also export subtitles
python transcribe.py interview.mp3 --model medium --language en --format all

# Timestamped transcript ([HH:MM:SS] before each line)
python transcribe.py lecture.m4a --timestamps

# Transcribe every audio/video file in a folder into ./out
python transcribe.py ./recordings --output ./out

# Improve accuracy for tricky names/terms
python transcribe.py call.wav --prompt "Dr. Nwosu, mitochondria, ATP synthase"
```

Options: `--model`, `--language`, `--output`, `--format {txt,srt,vtt,all}`,
`--timestamps`, `--prompt`, `--beam-size`, `--no-vad`. Run `python transcribe.py -h`
for the full list.

> On Windows you can also drag an audio file onto **`transcribe.bat`**.

## Choosing a model (accuracy vs. speed)

Bigger models are more accurate but slower — especially on CPU. Speed = roughly how much
faster than real time on a typical CPU.

| Model             | Accuracy      | Speed on CPU     | Download | Good for                     |
|-------------------|---------------|------------------|----------|------------------------------|
| `tiny`            | basic         | very fast        | ~75 MB   | quick drafts, clear speech   |
| `base`            | okay          | fast             | ~145 MB  | quick notes                  |
| `small` (default) | good          | ~2–4× real-time  | ~480 MB  | most everyday use            |
| `medium`          | very good     | ~real-time       | ~1.5 GB  | important / noisy audio      |
| `large-v3-turbo`  | excellent     | slower           | ~1.6 GB  | best accuracy, reasonable    |
| `large-v3`        | best          | slowest          | ~3 GB    | maximum accuracy, be patient |

Start with `small`. If a word is wrong or the audio is noisy/accented, rerun with
`medium` or `large-v3-turbo`, or pass a `--prompt` hint with the correct spellings.
Change the default with an env var: `WHISPER_MODEL=medium`.

## Deployment

The app reads a few environment variables so you can change how it's served without
editing code:

```bash
# Reachable from other devices on your network (visit http://<your-ip>:7860)
HOST=0.0.0.0 python app.py

# Create a temporary public https link (Gradio tunnel)
SHARE=1 python app.py

# Change the port
PORT=8080 python app.py
```

On Windows, set variables with `set HOST=0.0.0.0` on a separate line before `python app.py`.
To auto-start the UI at login on Windows, put a shortcut to `run-web.bat` in the
`shell:startup` folder.

## How it works

`core.py` wraps `faster-whisper`: it picks GPU/CPU automatically, loads (and caches) the
requested Whisper model, runs transcription with beam search and voice-activity
filtering, and formats the result as text, timestamped text, SRT, or VTT. Both `app.py`
(Gradio UI) and `transcribe.py` (CLI) are thin front-ends over that shared core.

## License

MIT — see [LICENSE](LICENSE).

---

*Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
[Gradio](https://www.gradio.app/). Runs entirely on your machine.*

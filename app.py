"""
Browser UI for the transcriber.

Run it with:
    python app.py
...then open the printed URL (default http://127.0.0.1:7860) in your browser.
Drag in an audio file (or record from your mic), pick a model, hit Transcribe.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

import core

# Label -> Whisper language code (None = auto-detect)
LANGUAGES = {
    "Auto-detect": None,
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Arabic": "ar", "Hindi": "hi", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "Turkish": "tr", "Polish": "pl", "Ukrainian": "uk",
    "Yoruba": "yo", "Swahili": "sw", "Hausa": "ha",
}

MODEL_HINTS = {
    "tiny": "fastest, least accurate",
    "base": "fast",
    "small": "good balance (default)",
    "medium": "more accurate, slower",
    "large-v3-turbo": "near-large accuracy, much faster than large",
    "large-v3": "most accurate, slowest on CPU",
}


def run(audio_path, model_size, language_label, include_timestamps, hint,
        progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Please upload an audio file or record from your microphone first.")

    language = LANGUAGES.get(language_label)

    def on_progress(done: float, total: float):
        frac = min(done / total, 1.0) if total else 0.0
        progress(frac, desc=f"Transcribing… {frac * 100:.0f}%")

    progress(0.0, desc=f"Loading '{model_size}' model (first use downloads it once)…")

    result = core.transcribe(
        audio_path,
        model_size=model_size,
        language=language,
        initial_prompt=(hint.strip() or None) if hint else None,
        progress=on_progress,
    )

    transcript = (
        core.to_timestamped_text(result.segments)
        if include_timestamps else result.text
    )

    # Write downloadable files into a temp folder Gradio can serve.
    out_dir = Path(tempfile.mkdtemp(prefix="transcript_"))
    txt = out_dir / "transcript.txt"
    srt = out_dir / "transcript.srt"
    vtt = out_dir / "transcript.vtt"
    txt.write_text(transcript + "\n", encoding="utf-8")
    srt.write_text(core.to_srt(result.segments), encoding="utf-8")
    vtt.write_text(core.to_vtt(result.segments), encoding="utf-8")

    info = (
        f"**Detected language:** {result.language} "
        f"({result.language_probability:.0%} confident)  \n"
        f"**Audio length:** {result.duration:.0f}s  ·  "
        f"**Model:** {result.model}  ·  **Segments:** {len(result.segments)}"
    )

    return transcript, info, [str(txt), str(srt), str(vtt)]


def build_ui() -> gr.Blocks:
    device, compute = core.pick_device()
    device_note = ("running on GPU" if device == "cuda"
                   else "running on CPU — larger models are slower")

    with gr.Blocks(title="Audio → Text Transcriber", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎧 Audio → Text Transcriber\n"
            f"Accurate, fully offline transcription with Whisper · *{device_note}*."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    type="filepath",
                    sources=["upload", "microphone"],
                    label="Audio (upload a file or record)",
                )
                model = gr.Dropdown(
                    choices=[(f"{m}  —  {MODEL_HINTS[m]}", m) for m in core.MODEL_CHOICES],
                    value=core.DEFAULT_MODEL,
                    label="Model (accuracy vs. speed)",
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Auto-detect",
                    label="Language",
                )
                hint = gr.Textbox(
                    label="Accuracy hint (optional)",
                    placeholder="Names / jargon likely in the audio, e.g. proper nouns…",
                    lines=1,
                )
                timestamps = gr.Checkbox(label="Include [timestamps] in the text", value=False)
                go = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=1):
                info = gr.Markdown()
                out_text = gr.Textbox(
                    label="Transcript",
                    lines=18,
                    buttons=["copy"],  # Gradio 6: adds a copy-to-clipboard button
                    placeholder="Your transcript will appear here…",
                )
                downloads = gr.File(label="Download (.txt, .srt, .vtt)", file_count="multiple")

        go.click(
            run,
            inputs=[audio, model, language, timestamps, hint],
            outputs=[out_text, info, downloads],
        )
        gr.Markdown(
            "First run of each model downloads it once (tiny ~75MB … large ~1.5GB), "
            "then it's cached for offline use."
        )
    return demo


if __name__ == "__main__":
    import os

    ui = build_ui()
    ui.queue()  # enables the live progress bar
    # These can be overridden with environment variables (see README > Deployment):
    #   HOST=0.0.0.0  -> reachable from other devices on your network
    #   PORT=8080     -> change the port
    #   SHARE=1       -> create a temporary public https link
    ui.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("SHARE", "").lower() in ("1", "true", "yes"),
        inbrowser=True,
    )

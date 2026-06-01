"""CLI entry point for the meeting-notes app."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from meeting_notes.config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OUTPUT_DIR,
    WHISPER_MODELS,
    AppConfig,
)
from meeting_notes.document import generate_document
from meeting_notes.summarizer import Summarizer
from meeting_notes.transcriber import Transcriber

console = Console()


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="meeting-notes")
def cli() -> None:
    """Meeting Notes AI – offline, multi-language meeting transcription and notes.

    \b
    Supported languages include English, Filipino/Tagalog, Spanish, French,
    Chinese, Japanese, and 90+ more (powered by OpenAI Whisper).

    \b
    Examples:

      # Record from microphone and generate notes
      meeting-notes record

      # Transcribe an existing audio file
      meeting-notes transcribe meeting.wav

      # Transcribe a Filipino meeting
      meeting-notes transcribe meeting.wav --language tl
    """


# ---------------------------------------------------------------------------
# record – live microphone recording
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--model",
    default="base",
    show_default=True,
    type=click.Choice(WHISPER_MODELS),
    help="Whisper model size (larger = more accurate, slower).",
)
@click.option(
    "--language",
    default=None,
    help='Language code, e.g. "en", "tl" (Filipino). Auto-detected if omitted.',
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory to save the meeting document.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    show_default=True,
    type=click.Choice(["markdown", "docx"]),
    help="Output document format.",
)
@click.option(
    "--no-ollama",
    is_flag=True,
    default=False,
    help="Skip Ollama and use the built-in extractive summarizer.",
)
@click.option(
    "--ollama-model",
    default=DEFAULT_OLLAMA_MODEL,
    show_default=True,
    help="Ollama model to use for summarization.",
)
@click.option(
    "--ollama-url",
    default=DEFAULT_OLLAMA_URL,
    show_default=True,
    help="Ollama server URL.",
)
def record(
    model: str,
    language: Optional[str],
    output_dir: str,
    output_format: str,
    no_ollama: bool,
    ollama_model: str,
    ollama_url: str,
) -> None:
    """Record audio from the microphone, transcribe, and generate meeting notes."""
    from meeting_notes.audio import AudioRecorder

    config = AppConfig(
        whisper_model=model,
        language=language,
        use_ollama=not no_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        output_dir=output_dir,
        output_format=output_format,
    )
    config.validate()

    console.print(
        Panel(
            Text.assemble(
                ("Meeting Notes AI\n", "bold cyan"),
                ("Press ", ""),
                ("Ctrl+C", "bold yellow"),
                (" to stop recording.", ""),
            ),
            border_style="cyan",
        )
    )

    recorder = AudioRecorder(
        sample_rate=config.sample_rate,
        channels=config.channels,
    )

    console.print("[green]● Recording…[/green]  (press Ctrl+C when done)")
    recorder.start()
    start_time = time.time()

    try:
        while True:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            console.print(
                f"  [dim]{mins:02d}:{secs:02d}[/dim]",
                end="\r",
                highlight=False,
            )
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    console.print("\n[yellow]■ Stopping recording…[/yellow]")
    audio_path = recorder.stop()
    console.print(f"[dim]Audio saved to: {audio_path}[/dim]")

    _process_audio(audio_path, config)


# ---------------------------------------------------------------------------
# transcribe – process an existing audio file
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, readable=True))
@click.option(
    "--model",
    default="base",
    show_default=True,
    type=click.Choice(WHISPER_MODELS),
    help="Whisper model size.",
)
@click.option(
    "--language",
    default=None,
    help='Language code, e.g. "en", "tl" (Filipino). Auto-detected if omitted.',
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory to save the meeting document.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    show_default=True,
    type=click.Choice(["markdown", "docx"]),
    help="Output document format.",
)
@click.option(
    "--no-ollama",
    is_flag=True,
    default=False,
    help="Skip Ollama and use the built-in extractive summarizer.",
)
@click.option(
    "--ollama-model",
    default=DEFAULT_OLLAMA_MODEL,
    show_default=True,
    help="Ollama model to use for summarization.",
)
@click.option(
    "--ollama-url",
    default=DEFAULT_OLLAMA_URL,
    show_default=True,
    help="Ollama server URL.",
)
def transcribe(
    audio_file: str,
    model: str,
    language: Optional[str],
    output_dir: str,
    output_format: str,
    no_ollama: bool,
    ollama_model: str,
    ollama_url: str,
) -> None:
    """Transcribe AUDIO_FILE and generate meeting notes."""
    config = AppConfig(
        whisper_model=model,
        language=language,
        use_ollama=not no_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        output_dir=output_dir,
        output_format=output_format,
    )
    config.validate()
    _process_audio(audio_file, config)


# ---------------------------------------------------------------------------
# languages – list supported languages
# ---------------------------------------------------------------------------


@cli.command()
def languages() -> None:
    """List languages supported by Whisper."""
    from meeting_notes.config import SUPPORTED_LANGUAGES

    console.print("[bold]Common supported languages:[/bold]")
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        if code == "auto":
            continue
        console.print(f"  [cyan]{code:<6}[/cyan] {name}")
    console.print(
        "\n[dim]Whisper supports 99+ languages in total. "
        "Pass any ISO 639-1 code with --language.[/dim]"
    )


# ---------------------------------------------------------------------------
# Shared processing pipeline
# ---------------------------------------------------------------------------


def _process_audio(audio_path: str, config: AppConfig) -> None:
    """Run the full transcription → summarization → document pipeline."""

    # 1. Transcribe
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Transcribing with Whisper ({config.whisper_model})…", total=None
        )
        transcriber = Transcriber(
            model_size=config.whisper_model,
            language=config.language,
        )
        try:
            result = transcriber.transcribe_file(audio_path)
        except Exception as exc:
            console.print(f"[red]Transcription failed:[/red] {exc}")
            sys.exit(1)

    console.print(
        f"[green]✓[/green] Transcription complete "
        f"([cyan]{result.language_name}[/cyan] detected)"
    )

    # 2. Summarize
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        backend = "Ollama" if config.use_ollama else "extractive"
        progress.add_task(f"Generating meeting notes ({backend})…", total=None)
        summarizer = Summarizer(
            use_ollama=config.use_ollama,
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model,
        )
        notes = summarizer.summarize(result.text, language=result.language)

    console.print("[green]✓[/green] Meeting notes generated")

    # 3. Build timed transcript
    timed = result.timed_transcript or result.text

    # 4. Write document
    doc_path = generate_document(
        notes=notes,
        transcript=timed,
        output_dir=config.output_dir,
        output_format=config.output_format,
    )

    console.print(f"[green]✓[/green] Document saved: [bold]{doc_path}[/bold]")

    # 5. Show a preview
    console.print()
    console.print(
        Panel(
            _preview(notes),
            title="[bold cyan]Meeting Notes Preview[/bold cyan]",
            border_style="cyan",
        )
    )


def _preview(notes) -> str:
    """Build a short preview string for display in the terminal."""
    lines = [f"[bold]{notes.title}[/bold]", "", notes.summary]
    if notes.action_items:
        lines.append("")
        lines.append("[bold]Action items:[/bold]")
        for item in notes.action_items[:5]:
            lines.append(f"  • {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()

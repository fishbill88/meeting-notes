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
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from meeting_notes.backends.router import BACKEND_NAMES
from meeting_notes.config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WHISPER_MODEL,
    WHISPER_MODELS,
    AppConfig,
    LanguageMode,
)
from meeting_notes.document import generate_document
from meeting_notes.pipeline import run_pipeline

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
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    type=click.Choice(WHISPER_MODELS),
    help="Whisper model size (larger = more accurate, slower).",
)
@click.option(
    "--language",
    default="auto",
    show_default=True,
    help=(
        'Language mode: "en", "tl" (Filipino/Tagalog), "ceb" (Cebuano/Bisaya), '
        'or "auto". "tl" routes to the Tagalog engine and "ceb" to the '
        "Omnilingual engine (~290 MB download on first use), not Whisper."
    ),
)
@click.option(
    "--backend",
    default=None,
    type=click.Choice(BACKEND_NAMES),
    help="Force a transcription engine instead of routing by language.",
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
    "--no-diarize",
    is_flag=True,
    default=False,
    help="Skip speaker identification (Person 1/2 labels).",
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
    backend: Optional[str],
    output_dir: str,
    output_format: str,
    no_diarize: bool,
    no_ollama: bool,
    ollama_model: str,
    ollama_url: str,
) -> None:
    """Record audio from the microphone, transcribe, and generate meeting notes."""
    from meeting_notes.audio import AudioRecorder

    config = _build_config(
        model=model,
        language=language,
        backend=backend,
        output_dir=output_dir,
        output_format=output_format,
        no_diarize=no_diarize,
        no_ollama=no_ollama,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )

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
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    type=click.Choice(WHISPER_MODELS),
    help="Whisper model size.",
)
@click.option(
    "--language",
    default="auto",
    show_default=True,
    help=(
        'Language mode: "en", "tl" (Filipino/Tagalog), "ceb" (Cebuano/Bisaya), '
        'or "auto". "tl" routes to the Tagalog engine and "ceb" to the '
        "Omnilingual engine (~290 MB download on first use), not Whisper."
    ),
)
@click.option(
    "--backend",
    default=None,
    type=click.Choice(BACKEND_NAMES),
    help="Force a transcription engine instead of routing by language.",
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
    "--no-diarize",
    is_flag=True,
    default=False,
    help="Skip speaker identification (Person 1/2 labels).",
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
    backend: Optional[str],
    output_dir: str,
    output_format: str,
    no_diarize: bool,
    no_ollama: bool,
    ollama_model: str,
    ollama_url: str,
) -> None:
    """Transcribe AUDIO_FILE and generate meeting notes."""
    config = _build_config(
        model=model,
        language=language,
        backend=backend,
        output_dir=output_dir,
        output_format=output_format,
        no_diarize=no_diarize,
        no_ollama=no_ollama,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )
    _process_audio(audio_file, config)


# ---------------------------------------------------------------------------
# languages – list supported languages
# ---------------------------------------------------------------------------


@cli.command()
def languages() -> None:
    """List languages supported by Whisper."""
    from meeting_notes.config import SUPPORTED_LANGUAGES

    console.print("[bold]Language modes:[/bold]")
    console.print("  [cyan]en    [/cyan] English            → faster-whisper")
    console.print("  [cyan]tl    [/cyan] Filipino / Tagalog → NeMo FastConformer")
    console.print("  [cyan]ceb   [/cyan] Cebuano / Bisaya   → Omnilingual CTC (Meta)")
    console.print("  [cyan]auto  [/cyan] Auto-detect        → faster-whisper")
    console.print("  [dim]taglish  Mixed EN/TL       → not implemented yet[/dim]")

    console.print("\n[bold]Other languages supported by Whisper:[/bold]")
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        if code in ("auto", "en", "tl", "fil", "ceb", "taglish"):
            continue
        console.print(f"  [cyan]{code:<6}[/cyan] {name}")
    console.print(
        "\n[dim]Whisper supports 99+ languages in total. Pass any ISO 639-1 "
        "code with --language to auto-detect through Whisper.[/dim]"
    )


# ---------------------------------------------------------------------------
# doctor – report which engines are installed and ready
# ---------------------------------------------------------------------------


@cli.command()
def doctor() -> None:
    """Check which transcription engines are installed and ready."""
    from meeting_notes.backends.router import available_backends
    from meeting_notes.backends.nemo import default_model_dir

    console.print("[bold]Transcription engines:[/bold]")
    status = available_backends()
    for name, ok in status.items():
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {mark} {name}")

    if not status.get("faster-whisper"):
        console.print(
            "\n[yellow]faster-whisper is missing.[/yellow] "
            "Install with: [bold]pip install -e .[/bold]"
        )
    if not status.get("nemo-tagalog"):
        console.print(
            f"\n[yellow]Tagalog engine is not set up.[/yellow] Expected model "
            f"files in [dim]{escape(str(default_model_dir()))}[/dim]\n"
            r"  1. [bold]pip install -e '.\[tagalog]'[/bold]" "\n"
            "  2. [bold]python scripts/export_nemo_to_onnx.py[/bold]\n"
            "[dim]Until then, --language tl falls back to faster-whisper.[/dim]"
        )


# ---------------------------------------------------------------------------
# vault – inspect and back up the consultation store
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--export-key",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Copy the encryption key to this path for safe keeping.",
)
@click.option(
    "--import-key",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Restore a previously exported key.",
)
@click.option(
    "--data-dir",
    default=None,
    help="Store location. Defaults to the app data directory.",
)
def vault(
    export_key: Optional[str],
    import_key: Optional[str],
    data_dir: Optional[str],
) -> None:
    """Inspect the consultation store, and back up its encryption key.

    \b
    Backing up meetings.db WITHOUT meetings.key produces a backup that can
    never be restored. Export the key once, keep it somewhere safe, and the
    database is recoverable.
    """
    import shutil
    from pathlib import Path as _Path

    from meeting_notes.storage import DB_FILENAME, KEY_FILENAME, default_data_dir

    directory = _Path(data_dir) if data_dir else default_data_dir()
    key_path = directory / KEY_FILENAME
    db_path = directory / DB_FILENAME

    if import_key:
        if key_path.exists():
            console.print(
                f"[red]A key already exists at[/red] {escape(str(key_path))}\n"
                "Move it aside first — overwriting it makes the current "
                "database unreadable."
            )
            sys.exit(2)
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(import_key, key_path)
        console.print(f"[green]✓[/green] Key restored to [bold]{escape(str(key_path))}[/bold]")
        return

    if export_key:
        if not key_path.exists():
            console.print(
                f"[red]No key found at[/red] {escape(str(key_path))}\n"
                "Nothing has been saved yet, or the store is in passphrase mode."
            )
            sys.exit(2)
        shutil.copy2(key_path, export_key)
        console.print(
            f"[green]✓[/green] Key exported to [bold]{escape(export_key)}[/bold]\n"
            "[yellow]Treat this file like the records themselves — anyone with "
            "it and a copy of the database can read every consultation.[/yellow]"
        )
        return

    # Default: report status.
    console.print("[bold]Consultation store[/bold]")
    console.print(f"  Location:  [dim]{escape(str(directory))}[/dim]")

    passphrase_mode = bool(os.environ.get("MEETING_NOTES_PASSPHRASE"))
    console.print(f"  Mode:      {'passphrase' if passphrase_mode else 'key-file'}")

    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        console.print(f"  Database:  [green]✓[/green] {size_kb:.0f} KB")
        try:
            from meeting_notes.storage import MeetingStore  # noqa: PLC0415

            store = MeetingStore(
                directory=directory,
                passphrase=os.environ.get("MEETING_NOTES_PASSPHRASE") or None,
            )
            console.print(f"  Records:   {store.count()}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  Records:   [red]unreadable[/red] — {escape(str(exc))}")
    else:
        console.print("  Database:  [dim]none yet[/dim]")

    if passphrase_mode:
        console.print(
            "\n[yellow]Passphrase mode: there is no recovery.[/yellow] "
            "A forgotten passphrase means the records are permanently unreadable."
        )
    elif key_path.exists():
        console.print(f"  Key:       [green]✓[/green] [dim]{escape(str(key_path))}[/dim]")
        console.print(
            "\n[yellow]Back this key up.[/yellow] A copy of the database "
            "without it cannot be restored:\n"
            "  [bold]meeting-notes vault --export-key backup.key[/bold]"
        )
    else:
        console.print("  Key:       [dim]not created yet[/dim]")


# ---------------------------------------------------------------------------
# bench – measure backends against a dataset
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--engine",
    "engines",
    multiple=True,
    default=("faster-whisper",),
    show_default=True,
    help=(
        "Engine to test, repeatable. Use 'name' or 'name:model' — e.g. "
        "'faster-whisper:medium', 'nemo-tagalog'."
    ),
)
@click.option(
    "--dataset",
    default=None,
    help="Named public dataset. Run 'bench --list' to see the options.",
)
@click.option(
    "--manifest",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="Local JSONL manifest instead of a public dataset.",
)
@click.option(
    "--language",
    default=None,
    help="Language of the manifest audio. Ignored for named datasets.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Only score the first N utterances. Use a small value to smoke-test.",
)
@click.option(
    "--terms",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="File of terms (one per line) to measure recall against.",
)
@click.option(
    "--output-dir",
    default="benchmarks/results",
    show_default=True,
    help="Where to write the markdown report.",
)
@click.option("--list", "list_datasets", is_flag=True, help="List datasets and exit.")
def bench(
    engines: tuple,
    dataset: Optional[str],
    manifest: Optional[str],
    language: Optional[str],
    limit: Optional[int],
    terms: Optional[str],
    output_dir: str,
    list_datasets: bool,
) -> None:
    """Measure transcription engines against a dataset.

    \b
    Examples:

      # Smoke test on 10 utterances
      meeting-notes bench --dataset fleurs-fil --limit 10

      # Compare both Tagalog options
      meeting-notes bench --dataset fleurs-fil \\
          --engine nemo-tagalog --engine faster-whisper:large-v3-turbo

      # Score your own audio
      meeting-notes bench --manifest benchmarks/taglish/manifest.jsonl
    """
    from datetime import datetime

    from meeting_notes.bench.datasets import (
        DATASET_SPECS,
        get_dataset,
        manifest_dataset,
    )
    from meeting_notes.bench.report import render_table, write_report
    from meeting_notes.bench.runner import run_benchmark

    if list_datasets:
        console.print("[bold]Available datasets:[/bold]")
        for name, spec in DATASET_SPECS.items():
            console.print(
                f"  [cyan]{name:<18}[/cyan] [dim]({spec['language']})[/dim] "
                f"{spec['description']}"
            )
        console.print(
            "\n[dim]Or use --manifest for local audio "
            "(JSONL: {\"audio\": ..., \"text\": ...}).[/dim]"
        )
        return

    if not dataset and not manifest:
        console.print(
            "[red]Nothing to benchmark.[/red] Pass --dataset or --manifest "
            "(or --list to see what is available)."
        )
        sys.exit(2)

    try:
        data = (
            manifest_dataset(manifest, language=language or "tl", limit=limit)
            if manifest
            else get_dataset(dataset, limit=limit)
        )
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        sys.exit(2)

    term_list = (
        [t.strip() for t in Path(terms).read_text(encoding="utf-8").splitlines() if t.strip()]
        if terms
        else []
    )

    console.print(
        f"[bold]Benchmarking[/bold] {data.name} "
        f"[dim]({data.description})[/dim]"
    )
    if data.download_mb:
        # Audio is embedded in one parquet file per split, so --limit does not
        # meaningfully shrink the fetch. Say so before it starts.
        console.print(
            f"[yellow]![/yellow] First run downloads ~{data.download_mb} MB "
            "and caches it. --limit does not reduce this."
        )
    console.print()

    runs = []
    for spec in engines:
        run = _run_one_engine(spec, data, term_list)
        if run is not None:
            runs.append(run)

    if not runs:
        console.print("[red]No engine produced results.[/red]")
        sys.exit(1)

    console.print()
    console.print(render_table(runs))

    stamp = datetime.now()
    path = write_report(
        runs,
        output_dir=output_dir,
        filename=f"{stamp:%Y-%m-%d}-{data.name}.md",
        title=f"Benchmark — {data.name}",
        date_str=f"{stamp:%Y-%m-%d %H:%M}",
    )
    console.print(f"[green]✓[/green] Report written: [bold]{escape(path)}[/bold]")


def _run_one_engine(spec: str, data, term_list: list):
    """Build and run a single engine spec like ``faster-whisper:medium``."""
    from meeting_notes.backends.router import resolve_backend
    from meeting_notes.bench.runner import run_benchmark

    name, _, model = spec.partition(":")
    config = AppConfig(
        whisper_model=model or DEFAULT_WHISPER_MODEL,
        language_mode=LanguageMode.from_value(data.language),
        backend=name,
    )
    try:
        config.validate()
        backend = resolve_backend(config)
    except (ValueError, NotImplementedError) as exc:
        console.print(f"[red]Skipping '{escape(spec)}':[/red] {escape(str(exc))}")
        return None

    label = spec if model else backend.name
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(f"{label}: loading…", total=None)

        def on_progress(index: int, result) -> None:
            progress.update(task, description=f"{label}: {index + 1} utterances")

        try:
            run = run_benchmark(
                backend, data, terms=term_list, on_progress=on_progress
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]'{escape(label)}' failed:[/red] {escape(str(exc))}")
            return None

    run.metrics.backend = label
    m = run.metrics
    console.print(
        f"[green]✓[/green] {label}: WER [bold]{m.wer * 100:.2f}%[/bold] "
        f"over {m.utterances} utterances"
        + (f" [red]({m.failures} failed)[/red]" if m.failures else "")
    )
    return run


# ---------------------------------------------------------------------------
# Shared processing pipeline
# ---------------------------------------------------------------------------


def _build_config(
    model: str,
    language: Optional[str],
    backend: Optional[str],
    output_dir: str,
    output_format: str,
    no_diarize: bool,
    no_ollama: bool,
    ollama_model: str,
    ollama_url: str,
) -> AppConfig:
    """Assemble and validate an :class:`AppConfig` from CLI options."""
    config = AppConfig(
        whisper_model=model,
        language_mode=LanguageMode.from_value(language),
        backend=backend,
        diarize=not no_diarize,
        use_ollama=not no_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        output_dir=output_dir,
        output_format=output_format,
    )
    try:
        config.validate()
    except ValueError as exc:
        console.print(f"[red]Invalid configuration:[/red] {escape(str(exc))}")
        sys.exit(2)
    return config


def _process_audio(audio_path: str, config: AppConfig) -> None:
    """Run the full transcription → summarization → document pipeline."""

    # 1. Transcribe + summarize (shared with the REST server)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Starting…", total=None)

        def on_progress(_stage: str, message: str) -> None:
            progress.update(task, description=message)

        try:
            result = run_pipeline(audio_path, config, on_progress=on_progress)
        except NotImplementedError as exc:
            console.print(f"[red]Not available yet:[/red] {escape(str(exc))}")
            sys.exit(2)
        except Exception as exc:
            # Backend hints contain pip extras like '.[tagalog]' — escape so
            # Rich renders them instead of parsing them as style tags.
            console.print(f"[red]Transcription failed:[/red] {escape(str(exc))}")
            sys.exit(1)

    console.print(
        f"[green]✓[/green] Transcription complete "
        f"([cyan]{result.transcript.language_name}[/cyan], "
        f"engine: [cyan]{result.backend}[/cyan])"
    )

    # The Tagalog router silently degrades to Whisper when the NeMo model has
    # not been exported. Say so rather than letting it pass unnoticed.
    if (
        config.language_mode is LanguageMode.TAGALOG
        and result.backend != "nemo-tagalog"
    ):
        console.print(
            "[yellow]![/yellow] Tagalog engine not installed — used "
            f"[cyan]{result.backend}[/cyan] instead. Run "
            "[bold]python scripts/export_nemo_to_onnx.py[/bold] for better "
            "Tagalog accuracy."
        )

    console.print("[green]✓[/green] Meeting notes generated")

    notes = result.notes

    # 2. Write document
    doc_path = generate_document(
        notes=notes,
        transcript=result.timed_transcript,
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
# serve – start the REST API server
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host. Loopback by default — this process handles patient audio.",
)
@click.option("--port", default=8811, show_default=True, help="Bind port.")
@click.option("--reload", is_flag=True, default=False, help="Auto-reload on code change.")
@click.option(
    "--token",
    default=None,
    help="Require this bearer token. Generated automatically unless --no-auth.",
)
@click.option(
    "--no-auth",
    is_flag=True,
    default=False,
    help="Disable token auth. Local development only.",
)
def serve(host: str, port: int, reload: bool, token: Optional[str], no_auth: bool) -> None:
    """Start the Meeting Notes REST API server."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        console.print(
            "[red]uvicorn is required.[/red] "
            "Install with: [bold]pip install 'meeting-notes\\[server]'[/bold]"
        )
        sys.exit(1)

    from meeting_notes.server import TOKEN_ENV, generate_token

    resolved = "" if no_auth else (token or generate_token())
    if resolved:
        os.environ[TOKEN_ENV] = resolved
    else:
        os.environ.pop(TOKEN_ENV, None)

    lines = [
        ("Meeting Notes API\n", "bold cyan"),
        (f"Listening on http://{host}:{port}\n", ""),
    ]
    if resolved:
        lines.append((f"Bearer token: {resolved}\n", "yellow"))
    else:
        lines.append(("Authentication DISABLED\n", "bold red"))
    if host not in ("127.0.0.1", "localhost", "::1"):
        lines.append(
            ("Reachable off this machine — do not do this with patient audio.\n", "bold red")
        )

    console.print(Panel(Text.assemble(*lines), border_style="cyan"))

    uvicorn.run(
        "meeting_notes.server:app",
        host=host,
        port=port,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()

"""Renders benchmark results as markdown.

Results go into ``benchmarks/results/`` so engine decisions have a dated,
reviewable paper trail rather than living in someone's terminal scrollback.

Every report states its normalisation settings and its hardware, because a WER
without those is not reproducible and an RTF without them is meaningless.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional, Sequence

from meeting_notes.bench.runner import BenchmarkRun


def format_percent(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def format_factor(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}×"


def hardware_summary() -> str:
    """One line describing the machine, for the report header."""
    return (
        f"{platform.system()} {platform.release()} / "
        f"{platform.machine()} / Python {platform.python_version()}"
    )


def render_table(runs: Sequence[BenchmarkRun]) -> str:
    """Render the comparison table — the part you actually read."""
    header = (
        "| Backend | Dataset | Utts | WER | CER | Term recall | RTF | vs real-time |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for run in runs:
        m = run.metrics
        rows.append(
            f"| `{m.backend}` | {m.dataset} | {m.utterances} | "
            f"{format_percent(m.wer)} | {format_percent(m.cer)} | "
            f"{format_percent(m.term_recall)} | "
            f"{'—' if m.rtf is None else f'{m.rtf:.3f}'} | "
            f"{format_factor(m.speed_factor)} |"
        )
    return header + "\n".join(rows) + "\n"


def render_report(
    runs: Sequence[BenchmarkRun],
    title: str = "Benchmark results",
    date_str: Optional[str] = None,
    notes: str = "",
    worst_n: int = 5,
) -> str:
    """Render the full markdown report."""
    lines: list[str] = [f"# {title}", ""]

    if date_str:
        lines += [f"**Date:** {date_str}", ""]
    lines += [f"**Hardware:** {hardware_summary()}", ""]

    lines += [
        "**Normalisation:** lowercase, punctuation stripped, whitespace "
        "collapsed, NFKC. Word-internal hyphens and apostrophes preserved so "
        "Tagalog affixed forms (`nag-meeting`) score as single tokens.",
        "",
        "**RTF** is compute-seconds per audio-second, measured after model "
        "warm-up. Below 1.0 is faster than real time.",
        "",
    ]

    if notes:
        lines += [notes, ""]

    lines += ["## Summary", "", render_table(runs)]

    failed = [r for r in runs if r.metrics.failures]
    if failed:
        lines += ["## Failures", ""]
        for run in failed:
            lines.append(
                f"- `{run.metrics.backend}` on {run.metrics.dataset}: "
                f"{run.metrics.failures} utterance(s) errored"
            )
        lines.append("")

    if worst_n:
        lines += ["## Worst utterances", ""]
        for run in runs:
            worst = run.worst(worst_n)
            if not worst:
                continue
            lines += [f"### `{run.metrics.backend}` — {run.metrics.dataset}", ""]
            for item in worst:
                lines += [
                    f"- **{item.id}** (WER {format_percent(item.wer)})",
                    f"  - ref: `{item.reference}`",
                    f"  - hyp: `{item.hypothesis}`",
                ]
            lines.append("")

    return "\n".join(lines)


def write_report(
    runs: Sequence[BenchmarkRun],
    output_dir: str = "benchmarks/results",
    filename: Optional[str] = None,
    **kwargs,
) -> str:
    """Write the report to ``output_dir`` and return its path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    name = filename or "latest.md"
    path = directory / name
    path.write_text(render_report(runs, **kwargs), encoding="utf-8")
    return str(path)

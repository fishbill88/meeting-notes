"""Local, encrypted storage for consultations.

Storage lives on the Python side rather than in Tauri so that one process owns
patient data end to end — the CLI, the server and the desktop app all read the
same records through the same encryption.

**What is stored, and what is not.** Record content (title, transcript, notes,
speaker names) is encrypted as a single payload. A little metadata stays in the
clear so the history list can sort and paginate without decrypting everything:
creation time, duration, language code, and which engine ran. None of that
identifies a patient.

**Audio is never persisted.** The desktop app holds a recording in memory,
uploads it, and the server deletes the temp file. There is no audio-retention
setting because there is no audio to retain — which is also the simplest
possible answer to a data-protection question about recordings.

**Search decrypts.** Encrypted content cannot be indexed by SQLite FTS without
leaking it, so :meth:`MeetingStore.search` decrypts rows and matches in memory.
That is linear in the number of records and comfortable into the low tens of
thousands; past that this needs a searchable-encryption design rather than a
bigger loop.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from meeting_notes.crypto import Vault

#: Bumped whenever the schema changes; see ``_MIGRATIONS``.
SCHEMA_VERSION = 2

DB_FILENAME = "meetings.db"
KEY_FILENAME = "meetings.key"


@dataclass
class MeetingRecord:
    """One saved consultation. Mirrors ``MeetingRecord`` in the TS clients."""

    id: str
    created_at: str
    duration_seconds: float
    transcript: dict
    notes: dict
    speaker_names: dict = field(default_factory=dict)
    #: The person's own notepad lines as they were when Enhance last ran: ``[{"t": seconds | None, "text"}]``.
    my_notes: list = field(default_factory=list)
    #: The enhanced notes tree (``enhance.enhance``), or ``{}`` before Enhance has run. Derived: it can
    #: always be regenerated from ``my_notes`` and the transcript.
    enhanced: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.notes.get("title") or "Untitled consultation")

    @property
    def language(self) -> str:
        return str(self.transcript.get("language") or "en")

    @property
    def backend(self) -> str:
        return str(self.transcript.get("backend") or "")

    @classmethod
    def new(
        cls,
        transcript: dict,
        notes: dict,
        duration_seconds: float = 0.0,
        speaker_names: Optional[dict] = None,
    ) -> "MeetingRecord":
        return cls(
            id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_seconds=duration_seconds,
            transcript=transcript,
            notes=notes,
            speaker_names=speaker_names or {},
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MeetingRecord":
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            duration_seconds=float(data.get("duration_seconds") or 0.0),
            transcript=data.get("transcript") or {},
            notes=data.get("notes") or {},
            speaker_names=data.get("speaker_names") or {},
            my_notes=data.get("my_notes") or [],
            enhanced=data.get("enhanced") or {},
        )

    def searchable_text(self) -> str:
        """Everything :meth:`MeetingStore.search` looks through."""
        parts = [
            self.title,
            str(self.notes.get("summary") or ""),
            str(self.transcript.get("text") or ""),
        ]
        for key in ("key_topics", "decisions", "action_items"):
            parts.extend(str(v) for v in (self.notes.get(key) or []))
        parts.extend(str(line.get("text") or "") for line in self.my_notes if isinstance(line, dict))
        return "\n".join(parts)


@dataclass
class MeetingSummary:
    """List-view row. Decrypted, but without the full transcript."""

    id: str
    created_at: str
    duration_seconds: float
    language: str
    backend: str
    title: str
    preview: str


#: Applied in order for databases below :data:`SCHEMA_VERSION`.
_MIGRATIONS: list[str] = [
    # v1 — initial schema
    """
    CREATE TABLE IF NOT EXISTS meetings (
        id                TEXT PRIMARY KEY,
        created_at        TEXT NOT NULL,
        duration_seconds  REAL NOT NULL DEFAULT 0,
        language          TEXT NOT NULL DEFAULT '',
        backend           TEXT NOT NULL DEFAULT '',
        payload           BLOB NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_meetings_created
        ON meetings (created_at DESC);
    """,
    # v2 - notepads: what a person types during (or after) a call, kept under a key the host chooses so it
    # survives before any notes exist. Encrypted like a meeting.
    """
    CREATE TABLE IF NOT EXISTS notepads (
        key         TEXT PRIMARY KEY,
        updated_at  TEXT NOT NULL,
        payload     BLOB NOT NULL
    );
    """,
]


class MeetingStore:
    """SQLite-backed, encrypted consultation store.

    Args:
        directory: Where the database and key live. Defaults to the app's
            data directory.
        passphrase: Enables passphrase mode on the vault.
    """

    def __init__(
        self,
        directory: Optional[Path] = None,
        passphrase: Optional[str] = None,
    ) -> None:
        self.directory = Path(directory) if directory else default_data_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / DB_FILENAME
        self.vault = Vault(self.directory / KEY_FILENAME, passphrase=passphrase)
        self._migrate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(self, record: MeetingRecord) -> str:
        """Insert or replace a record. Returns its id."""
        payload = self.vault.encrypt(json.dumps(record.to_dict()))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meetings
                    (id, created_at, duration_seconds, language, backend, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created_at = excluded.created_at,
                    duration_seconds = excluded.duration_seconds,
                    language = excluded.language,
                    backend = excluded.backend,
                    payload = excluded.payload
                """,
                (
                    record.id,
                    record.created_at,
                    record.duration_seconds,
                    record.language,
                    record.backend,
                    payload,
                ),
            )
        return record.id

    def get(self, record_id: str) -> Optional[MeetingRecord]:
        """Fetch one record, or None when it does not exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM meetings WHERE id = ?", (record_id,)
            ).fetchone()
        if row is None:
            return None
        return self._decode(row["payload"])

    def list(self, limit: int = 50, offset: int = 0) -> list[MeetingSummary]:
        """Most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, duration_seconds, language, backend, payload
                FROM meetings
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._summarize(row) for row in rows]

    def search(self, query: str, limit: int = 50) -> list[MeetingSummary]:
        """Case-insensitive substring search across titles, notes and transcripts.

        Decrypts as it scans — see the module docstring on why, and on the
        scale at which that stops being reasonable.
        """
        needle = query.strip().lower()
        if not needle:
            return self.list(limit=limit)

        results: list[MeetingSummary] = []
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, created_at, duration_seconds, language, backend, payload
                FROM meetings
                ORDER BY created_at DESC
                """
            )
            for row in cursor:
                record = self._decode(row["payload"])
                if needle in record.searchable_text().lower():
                    results.append(self._summarize(row, record))
                    if len(results) >= limit:
                        break
        return results

    def delete(self, record_id: str) -> bool:
        """Delete a record. Returns whether it existed."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM meetings WHERE id = ?", (record_id,))
            return cursor.rowcount > 0

    # -- notepads ------------------------------------------------------

    def save_notepad(self, key: str, lines: list[dict]) -> str:
        """Replace the notepad stored under ``key``. Returns its update time (UTC, ISO)."""
        updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = self.vault.encrypt(json.dumps({"lines": lines}))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notepads (key, updated_at, payload) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET updated_at = excluded.updated_at, payload = excluded.payload
                """,
                (key, updated, payload),
            )
        return updated

    def get_notepad(self, key: str) -> Optional[tuple[list[dict], str]]:
        """``(lines, updated_at)`` for ``key``, or ``None`` when nothing was ever saved there."""
        with self._connect() as conn:
            row = conn.execute("SELECT updated_at, payload FROM notepads WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(self.vault.decrypt(row["payload"])).get("lines") or [], row["updated_at"]

    def delete_notepad(self, key: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM notepads WHERE key = ?", (key,)).rowcount > 0

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM meetings").fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self) -> None:
        """Bring the database up to :data:`SCHEMA_VERSION`.

        ``user_version`` is used rather than a table so the very first
        migration has somewhere to record its own completion.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            for index in range(current, len(_MIGRATIONS)):
                conn.executescript(_MIGRATIONS[index])
                conn.execute(f"PRAGMA user_version = {index + 1}")
            conn.commit()

    def _decode(self, payload: bytes) -> MeetingRecord:
        return MeetingRecord.from_dict(json.loads(self.vault.decrypt(payload)))

    def _summarize(
        self,
        row: sqlite3.Row,
        record: Optional[MeetingRecord] = None,
    ) -> MeetingSummary:
        record = record or self._decode(row["payload"])
        summary_text = str(record.notes.get("summary") or "")
        return MeetingSummary(
            id=row["id"],
            created_at=row["created_at"],
            duration_seconds=float(row["duration_seconds"]),
            language=row["language"],
            backend=row["backend"],
            title=record.title,
            preview=_truncate(summary_text, 180),
        )


def default_data_dir() -> Path:
    """Where consultations live.

    Overridden by ``MEETING_NOTES_DATA_DIR`` so the desktop shell can point
    this at its Tauri app-data directory.
    """
    import os

    override = os.environ.get("MEETING_NOTES_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".meeting-notes" / "data"


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def summary_to_dict(summary: MeetingSummary) -> dict[str, Any]:
    """Wire format for the history list."""
    return asdict(summary)
